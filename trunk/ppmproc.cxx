/*
 * ppmproc.cxx v. 0.1
 *
 * Copyright (C) 2009 Chad Page 
 *
 * Licensed under LGPL, V2 or later
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <math.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sched.h>
#include <vector>
#include <pthread.h>

struct Point
{
	int x, y;
	
	int inline a() {return x * y;}
	bool inline isValid(int px, int py) {
		return ((px < 0) || (py < 0) || (px > x) || (py > y)) ? false : true;
	}
	bool inline isValid(Point p) {
		return isValid(p.x, p.y);
	}
	Point(int _x = 0, int _y = 0) {x = _x; y = _y;}
//	Point(Point &p) {x = p.x; y = p.y;}
};

struct Rect
{
	Point tl, br;
	bool inline isValid(Rect r);
	Rect(Point _tl = Point(0, 0), Point _br = Point(0,0)) {tl = _tl; br = _br;}
};

// Could use a union here, but shorter not to.  
// Convention:  y/r/b for y/cr/cb, and y/r/g/b.  y is either 255 or 1.0 in pure rgb mode.

template<class T> struct HSL;

template<class T> struct RGBY
{
	T y;	// MUST be first
	T r;
	T g;
	T b;

	RGBY FromHSL(HSL<T> old) {
		RGBY c;
		T t[3], q, p;
		int i;

		if (old.s <= 0.0) {c.r = c.g = c.b = old.l; return c;}

		old.l = clip(old.l);

		t[1] = fmod(old.h, 360.0) / 360.0; // g
		t[0] = t[1] + (1.0/3.0); // r
		t[2] = t[1] - (1.0/3.0); // b
//		fprintf(stderr, "t %2f,%2f,%2f\n", t[0], t[1], t[2]);

		if (old.l < 0.5)
			q = old.l * (1.0 + old.s);
		else
			q = (old.l + old.s) - (old.l * old.s);

		p = (2 * old.l) - q;

		for (i = 0; i <= 2; i++) {
			while (t[i] < 0.0) t[i] += 1.0;
			while (t[i] > 1.0) t[i] -= 1.0;

			if ((t[i] * 6) < 1.0) t[i] = p + (q - p) * 6.0 * t[i];
			else if (t[i] < 0.5) t[i] = q;
			else if ((3 * t[i]) < 2) t[i] = p + (q - p) * (6.0 * ((2.0 / 3.0) - t[i]));
			else t[i] = p;
		}
		c.r = t[0]; c.g = t[1]; c.b = t[2];

//		fprintf(stderr, "-> %2f,%2f,%2f\n", c.r, c.g, c.b);

		return c;
	}	

	void max(RGBY<T> &a, RGBY<T> &b)
	{
		y = (a.y > b.y) ? a.y : b.y;
		r = (a.r > b.r) ? a.r : b.r;
		g = (a.g > b.g) ? a.g : b.g;
		b = (a.b > b.b) ? a.b : b.b;
	}
};

template<class T> struct HSL
{
	T l;	// MUST be first
	T s;
	T h;
	T filler;

	HSL FromRGB(RGBY<T> &old) {
		HSL c;
		T min, max, hbase;
		int colmax = 1;

//		fprintf(stderr, "r%2f,%2f,%2f ", old.r, old.g, old.b);

		// get min/max and max color
		min = old.g; max = old.g; 
		if (old.b < min) min = old.b;
		if (old.r < min) min = old.r;
		if (old.b > max) {max = old.b; colmax = 2;}
		if (old.r > max) {max = old.r; colmax = 3;}

		c.l = (max + min) / 2.0;
		if (!c.l) {
			c.s = 0;
			c.h = 0;
			return c;
		}

//		fprintf(stderr, "%lf %lf ", min, max);

		c.s = (max - min) / ((c.l > 0.5) ? (2 - (2 * c.l)) : (2 * c.l));

		if (colmax == 1) hbase = old.b - old.r;
		else if (colmax == 2) hbase = old.r - old.g;
		else if (colmax == 3) hbase = old.g - old.b;
		
		c.h = ((60.0 * hbase) / (max - min)) + (colmax * 120.0);
//		fprintf(stderr, "%f %f %f %d %f\n", old.r, old.g, old.b, colmax, hbase);
		c.h = fmodf(c.h, 360.0);

//		fprintf(stderr, "-> %2f,%2f,%2f\n", c.h, c.s, c.l);

		return c;
	}
};


template<class T> class Pixel
{
public:
	union {
		T color[4];
		RGBY<T> rgb;
		HSL<T> hsl;
	};
};

template<class T>
T clip(T in) 
{
	if (in < 0.0) in = 0.0;
	else if (in > 1.0) in = 1.0;
	
	return in;
}

#define M_PI 3.14159265358979323846

template<class T>
void prepDCTtable(T *ctab, int size, bool inverse = false)
{
	int bin, k;

	for (bin = 0; bin < size; bin++) { 
		T core = (bin + (inverse ? 0.0 : 0.5)) * M_PI / size;
    		for (k = 0; k < size; k++) { 
        		ctab[(k * size) + bin] = cos((k + (inverse ? 0.5: 0.0)) * core);
		}
	}
}

template<class T>
struct DCTqueue {
	T *out, *in, *arg;
	int size;
	bool inverse;

	DCTqueue() {set();}
	void set(T *_out = NULL, T *_in = NULL, T *_arg = NULL, int _size = 0, bool _inverse = false) {
		out = _out; in = _in; arg = _arg; size = _size; inverse = _inverse;
	}
	void run() {
//		fprintf(stderr, "%x %x %x %d %d\n", out, in, arg, size, inverse);
		DCT(out, in, arg, size, inverse);
//		fprintf(stderr, "r");
	}
};

// this routine's been (re)written to set things up so vector accesses are aligned
// in the main loop.  this buys about 30% on a C2D.

template<class T>
void DCT(T *out, T *in, T *arg, int size, bool inverse = false)
{
	int n, k;
	float vin[8192]__attribute__((aligned(16))), vin0 = 0.0;
	float varg[8192]__attribute__((aligned(16)));
	float sqrt2 = sqrt(2.0);
	float sqrt2s = sqrt(2.0 / size);

	for (k = 0; k < size; k++)
		vin[k] = in[k];

	if (inverse) {
		vin0 = in[0] * sqrt2 * (float)0.5;
		vin[0] = 0.0;
	}

    	for (k = 0; k < size; k++) { 
		T ndct = vin0;
		for (n = 0; n < size; n++) {
			varg[n] = arg[(k * size) + n];
		}
		for (n = 0; n < size; n++) { 
		        ndct += vin[n] * varg[n];
    		}
		if (!inverse && !k) ndct /= sqrt2;
		out[k] = ndct * sqrt2s;
	}

	return;
}

// Why is this so parallel?  Because it goes fast.  And array's don't auto-vectorize as well

// this is 35% faster than a quadDCT, which is ~30% faster than [single]DCT

template<class T>
void octoDCT(DCTqueue<T> param[])
{
	int n, k;
	int size = param[0].size;
	float varg[8192]__attribute__((aligned(16)));
	float vin1[8192]__attribute__((aligned(16))), vin1_0 = 0.0;
	float vin2[8192]__attribute__((aligned(16))), vin2_0 = 0.0;
	float vin3[8192]__attribute__((aligned(16))), vin3_0 = 0.0;
	float vin4[8192]__attribute__((aligned(16))), vin4_0 = 0.0;
	float vin5[8192]__attribute__((aligned(16))), vin5_0 = 0.0;
	float vin6[8192]__attribute__((aligned(16))), vin6_0 = 0.0;
	float vin7[8192]__attribute__((aligned(16))), vin7_0 = 0.0;
	float vin8[8192]__attribute__((aligned(16))), vin8_0 = 0.0;
	float sqrt2 = sqrt(2.0);
	float sqrt2s = sqrt(2.0 / size);

//	fprintf(stderr, "%d\n", size);

	for (k = 0; k < param[0].size; k++) {
		vin1[k] = param[0].in[k];
		vin2[k] = param[1].in[k];
		vin3[k] = param[2].in[k];
		vin4[k] = param[3].in[k];
		vin5[k] = param[4].in[k];
		vin6[k] = param[5].in[k];
		vin7[k] = param[6].in[k];
		vin8[k] = param[7].in[k];
	}

	if (param[0].inverse) {
		vin1_0 = vin1[0] * sqrt2 * (float)0.5;
		vin1[0] = 0.0;
		vin2_0 = vin2[0] * sqrt2 * (float)0.5;
		vin2[0] = 0.0;
		vin3_0 = vin3[0] * sqrt2 * (float)0.5;
		vin3[0] = 0.0;
		vin4_0 = vin4[0] * sqrt2 * (float)0.5;
		vin4[0] = 0.0;
		vin5_0 = vin5[0] * sqrt2 * (float)0.5;
		vin5[0] = 0.0;
		vin6_0 = vin6[0] * sqrt2 * (float)0.5;
		vin6[0] = 0.0;
		vin7_0 = vin7[0] * sqrt2 * (float)0.5;
		vin7[0] = 0.0;
		vin8_0 = vin8[0] * sqrt2 * (float)0.5;
		vin8[0] = 0.0;
	}

    	for (k = 0; k < size; k++) { 
		T ndct1 = vin1_0;
		T ndct2 = vin2_0;
		T ndct3 = vin3_0;
		T ndct4 = vin4_0;
		T ndct5 = vin5_0;
		T ndct6 = vin6_0;
		T ndct7 = vin7_0;
		T ndct8 = vin8_0;
		for (n = 0; n < size; n++) {
			varg[n] = param[0].arg[(k * size) + n];
		}
		for (n = 0; n < size; n++) { 
		        ndct1 += vin1[n] * varg[n];
		        ndct2 += vin2[n] * varg[n];
		        ndct3 += vin3[n] * varg[n];
		        ndct4 += vin4[n] * varg[n];
		        ndct5 += vin5[n] * varg[n];
		        ndct6 += vin6[n] * varg[n];
		        ndct7 += vin7[n] * varg[n];
		        ndct8 += vin8[n] * varg[n];
    		}
		if (!param[0].inverse && !k) {
			ndct1 /= sqrt2;
			ndct2 /= sqrt2;
			ndct3 /= sqrt2;
			ndct4 /= sqrt2;
			ndct5 /= sqrt2;
			ndct6 /= sqrt2;
			ndct7 /= sqrt2;
			ndct8 /= sqrt2;
		}
		param[0].out[k] = ndct1 * sqrt2s;
		param[1].out[k] = ndct2 * sqrt2s;
		param[2].out[k] = ndct3 * sqrt2s;
		param[3].out[k] = ndct4 * sqrt2s;
		param[4].out[k] = ndct5 * sqrt2s;
		param[5].out[k] = ndct6 * sqrt2s;
		param[6].out[k] = ndct7 * sqrt2s;
		param[7].out[k] = ndct8 * sqrt2s;
	}

	return;
}

template<class T>
void DCTnc(T *out, T *in, T *arg, int size, bool inverse = false)
{
	int px, pbin, i, j;
	int n, k;
	float sqrt2 = sqrt(2.0), sqrt2s(2.0 / size);
//	struct timespec t1, t2;

	if (inverse) in[0] *= sqrt(2.0);
    	for (k = 0; k < size; k++) { 
		T ndct = inverse ? (in[0] * 0.5) : 0.0;
		int abase = k * size;
		for (n = inverse ? 1: 0; n < size; n++) { 
		        ndct += in[n] * arg[abase + n];
    		}
		if (!inverse && !k) ndct /= sqrt2;
		out[k] = ndct * sqrt2s;
	}
//	clock_gettime(CLOCK_MONOTONIC, &t2);
//	fprintf(stderr, "%d\n", ((t2.tv_sec - t1.tv_sec) * 1000000000) + t2.tv_nsec - t1.tv_nsec);

	return;
}

void * DCTqueue_run(void *pqueue)
{
	int i, j, num = 0;
	DCTqueue<float> *queue = (DCTqueue<float> *)pqueue;
//	struct timespec t1, t2;
//	int qlen;

	while (queue++->out) num++;
	queue = (DCTqueue<float> *)pqueue;

	for (i = 0; (i + 7) < num; i+=8) {
		DCTqueue<float> array[8];
		
		for (j = 0; j < 8; j++) array[j] = queue[i + j];
//		fprintf(stderr, "%d %x %x\n", i + j, array[0]->size, pqueue);
		octoDCT((DCTqueue<float> *)array);
	}

	for (; i < num; i++) {
		queue[i].run();
	}
	return NULL;
}

template<class T> class Picture {
	protected:
		Point size;
		Pixel<T> *p;
	public:
		Picture(Point nSize, unsigned short int *picture = NULL) {
			size = nSize;
			p = new Pixel<T>[size.a()];
	
			if (!picture) return;
			Import(picture);
		}

		Picture(Picture &orig, int rotate = 0) {
			int i;

			if (rotate) {
				size.x = orig.size.y; size.y = orig.size.x;
			} else size = orig.size;

			p = new Pixel<T>[size.x * size.y];

			if (rotate) {
				int px, py;
				for (px = 0; px < size.x; px++) {
					for (py = 0; py < size.y; py++) {
						Set(px, py, orig.Get(py, px));
					}	
				}
			} else {
				for (i = 0; i < size.a(); i++) p[i] = orig.p[i];
			}
		}

		void Import(unsigned short int *picture = NULL)
		{
			int i;

			if (!picture) return;
			for (i = 0; i < size.a(); i++) {
				p[i].rgb.r = (T)picture[(i * 3)] / 65535.0;
				p[i].rgb.g = (T)picture[(i * 3) + 1] / 65535.0;
				p[i].rgb.b = (T)picture[(i * 3) + 2] / 65535.0;
			}
		}

		void Export(unsigned short int *picture = NULL)
		{
			int i;

			if (!picture) return;
			for (i = 0; i < size.a(); i++) {
				if (p[i].rgb.r < 0) p[i].rgb.r = 0;
				if (p[i].rgb.g < 0) p[i].rgb.g = 0;
				if (p[i].rgb.b < 0) p[i].rgb.b = 0;
				picture[(i * 3)] = (unsigned short int)(p[i].rgb.r * 65535.0);
				picture[(i * 3) + 1] = (unsigned short int)(p[i].rgb.g * 65535.0);
				picture[(i * 3) + 2] = (unsigned short int)(p[i].rgb.b * 65535.0);
				//fprintf(stderr, "%d %lf %lf %lf\n", i, p[i].rgb.r, p[i].rgb.g, p[i].rgb.b);
//				fprintf(stderr, "%d %d %d %d\n", i, picture[i * 3], picture[(i * 3) + 1], picture[(i * 3) + 2]);
			}
		}

		void Normalize() {
/*			int i;
			T m;
			for (i = 0; i < size.a(); i++) {
				if ((m = (p[i].rgb.y * p[i].rgb.r)) > 1.0) p[i].rgb.y /= m;
				if ((m = (p[i].rgb.y * p[i].rgb.g)) > 1.0) p[i].rgb.y /= m;
				if ((m = (p[i].rgb.y * p[i].rgb.b)) > 1.0) p[i].rgb.y /= m;
			}
*/		}

		inline bool Set(Point l, Pixel<T> c) {
			if (!size.isValid(l)) return false;
			p[(l.y * size.x) + l.x] =  c;
			return true;
		}

		inline bool Set(int x, int y, Pixel<T> c) {return Set(Point(x, y), c);}

		// returns all 0 if out of bounds.  Throw exception instead?
		inline Pixel<T> Get(Point l) {
			if (!size.isValid(l)) return Pixel<T>();
			return p[(l.y * size.x) + l.x];
		}

		inline Pixel<T> Get(int x, int y) {return Get(Point(x, y));}

		inline Point Crop(Point t) {
			Point np = t;

			if (np.x < 0) np.x = 0;
			if (np.y < 0) np.y = 0;
			if (np.x >= size.x) np.x = size.x;
			if (np.y >= size.y) np.y = size.y;

			return np;
		}

		inline T GetAvg() {
			T avg = 0;
			int px, py;

			for (py = 0; py < size.y; py++) {
				for (px = 0; px < size.x; px++) {
					avg += p[(py * size.x) + px].rgb.y;
				}
			}	
			avg /= size.a();
			return avg;
		}

		void RGBtoYRGB(Picture &o) {
			int i;

			for (i = 0; i < size.a(); i++) {
//				p[i].rgb.y = (o.p[i].rgb.r * 299 / 1000) + (o.p[i].rgb.g * 587 / 1000) + (o.p[i].rgb.b * 114 / 1000);
				p[i].rgb.y = (o.p[i].rgb.r * .299) + (o.p[i].rgb.g * .587) + (o.p[i].rgb.b * .114);
				p[i].rgb.r = o.p[i].rgb.r / p[i].rgb.y;
				p[i].rgb.g = o.p[i].rgb.g / p[i].rgb.y;
				p[i].rgb.b = o.p[i].rgb.b / p[i].rgb.y;
			}
		}

		void RGBtoHSL(Picture &o) {
			int i;

			for (i = 0; i < size.a(); i++) {
				p[i].hsl = p[i].hsl.FromRGB(o.p[i].rgb);
			}
		}

		void HSLtoRGB(Picture &o) {
			int i;

			for (i = 0; i < size.a(); i++) {
				p[i].rgb = o.p[i].rgb.FromHSL(o.p[i].hsl);
			}
		}

		void Boost(Picture &o, Picture &n, T bright, T contrast, T color)
		{
			int i;
			T tot = 0.0, max = 0.0, bmax = 0.0, t;
			T gain, avg, peak;
			Picture<T> *tmpn;

			for (i = 0; i < size.a(); i++) {
				T factor = 0, tfactor = 0;

//				T mult = n.p[i].hsl.l / o.p[i].hsl.l;
				n.p[i].hsl.s *= color;
				if (n.p[i].hsl.s > 1.0) n.p[i].hsl.s = 1.0;
				if (n.p[i].hsl.s < 0.0) n.p[i].hsl.s = 0.0;
				tot += n.p[i].hsl.l;
				if (n.p[i].hsl.l > max) max = n.p[i].hsl.l;
			}
			avg = tot / size.a();

			fprintf(stderr, "avg %f max %f adjmax %f bright %f contrast %f\n", tot, max, max * (bright / avg), bright, contrast);
			peak = max * (bright / avg);
			max = 0.0;

			for (i = 0; i < size.a(); i++) {
				n.p[i].hsl.l *= (bright / avg);
				if (n.p[i].hsl.l > bmax) bmax = n.p[i].hsl.l;
				if (n.p[i].hsl.l < 0) n.p[i].hsl.l = 0;
			}

			tot = 0;
			for (i = 0; i < size.a(); i++) {
				if (n.p[i].hsl.l > bright) {
					t = 1 - ((n.p[i].hsl.l - bright) / (peak - bright));
//					fprintf(stderr, "%f %f ", n.p[i].hsl.l, t);
					n.p[i].hsl.l = ((1 - pow(t, contrast)) * (1 - bright)) + bright;
//					fprintf(stderr, "%f\n", n.p[i].hsl.l);
				}
				
				if (n.p[i].hsl.l > max) max = n.p[i].hsl.l;
				if (n.p[i].hsl.l > 1.0) n.p[i].hsl.l = 1.0;
				tot += n.p[i].hsl.l;
			}

			tmpn = new Picture<T>(n);

			for (i = 0; i < size.a(); i++) {
				int px = i % size.x, py = i / size.x, tx, R = 4;
				T lavg = 0, factor = 0, tfactor = 0;

	                       	for (tx = ((px - R) > 0) ? (px - R) : 0; (tx < (size.x - 1)) &&         ((tx - px) < R); tx++) {
                                	factor =  1.0 / ((abs(tx - px)) + 1);
                                	lavg += (tmpn->Get(tx, py).hsl.l * factor);
                                	tfactor += factor;
                        	}
//				fprintf(stderr, "%d, %d %f %f %f %f ", px, py, bmax, tfactor, lavg / tfactor, n.p[i].hsl.l);
				n.p[i].hsl.l += ((tmpn->p[i].hsl.l - (lavg / tfactor)) * tmpn->p[i].hsl.l);
//				fprintf(stderr, "%f\n", n.p[i].hsl.l);
				if (n.p[i].hsl.l > peak) peak = n.p[i].hsl.l;
				if (n.p[i].hsl.l < 0.0) n.p[i].hsl.l = 0;
				if (n.p[i].hsl.l > 1.0) n.p[i].hsl.l = 1.0;
			}

			delete tmpn;

			fprintf(stderr, "%f %f\n", max, tot / size.a());
		}

		void YRGBtoRGB(Picture &o) {
			int i;

			for (i = 0; i < size.a(); i++) {
				p[i].rgb.r = o.p[i].rgb.r * o.p[i].rgb.y;
				p[i].rgb.g = o.p[i].rgb.g * o.p[i].rgb.y;
				p[i].rgb.b = o.p[i].rgb.b * o.p[i].rgb.y;
				p[i].rgb.y = 1.0;
//				fprintf(stderr, "%d %lf %lf %lf %lf\n", i, o.p[i].rgb.y, p[i].rgb.r, p[i].rgb.g, p[i].rgb.b);
			}
		}

		void hDCT(Picture &o, bool inverse, int field, T str = 0.0) {
			int bin;
			int i, px = 0, py = 0;
			T *dct, *row, *ctab;
			Pixel<T> tmp;

			double etime;
			double iter;
			struct timespec t1, t2;

			dct = new T[size.x];
			row = new T[size.x];
			ctab = (T *)memalign(16, size.x * size.x * sizeof(T));

			clock_gettime(CLOCK_MONOTONIC, &t1);
//	fprintf(stderr, "%d\n", ((t2.tv_sec - t1.tv_sec) * 1000000000) + t2.tv_nsec - t1.tv_nsec);

			prepDCTtable(ctab, size.x, inverse);

			for (py = 0; py < size.y; py++) {
				int i;

				for (bin = 0; bin < size.x; bin++) { 
					row[bin] = o.Get(bin, py).color[field];
				}
				DCTnc(dct, row, ctab, size.x, inverse);
				fprintf(stderr, "%d\n", py);

				if (!inverse) {
					p[(py * size.x)].y = dct[0];
					for (px = 1; px < size.x; px++) {
						float m = (1 + (((float)px / (float)size.x) * str));
						if (m <= 0) m = 0;
						p[(py * size.x) + px].y = dct[px] * m;
					}
				} else {
					for (px = 0; px < size.x; px++) {
						p[(py * size.x) + px].y = dct[px] * (str + 1.0);
					}
				}
			}
			
			clock_gettime(CLOCK_MONOTONIC, &t2);
			etime = ((t2.tv_sec - t1.tv_sec) * 1000000000) + t2.tv_nsec - t1.tv_nsec;
			iter = (size.x * size.x) * size.y;
			fprintf(stderr, "time:%lf nsec flops:%lf gflops\n", etime, iter / etime);

			delete dct; delete row; free(ctab);

			return;

		}

		void thDCT(Picture &o, bool inverse, int color = 0, T str = 0.0, int threads = 2) {
			int bin;
			int i, px = 0, py = 0;
			T **dct, **row, *ctab;
			DCTqueue<T> **queue;
			pthread_t pid[32];
			struct timespec t1, t2;
			double etime, iter;
		
			clock_gettime(CLOCK_MONOTONIC, &t1);

			if (threads > 32) threads = 32;

			dct = new T*[size.y];
			row = new T*[size.y];
			for (py = 0; py < size.y; py++) {
				dct[py] = (T *)memalign(16, size.x * sizeof(T));
				row[py] = (T *)memalign(16, size.x * sizeof(T));
			}
			ctab = (T *)memalign(16, size.x * size.x * sizeof(T));

			prepDCTtable(ctab, size.x, inverse);

			for (py = 0; py < size.y; py++) {
				for (bin = 0; bin < size.x; bin++) { 
					row[py][bin] = o.Get(bin, py).color[0];
				}
			}

			queue = new DCTqueue<T>*[threads];
			for (i = 0; i < threads; i++) {
				fprintf(stderr, "%d %d\n", i, threads);
				queue[i] = new DCTqueue<T>[(size.y / threads) + 2];
			}

			for (py = 0; py < size.y; py++) {
				queue[py % threads][py / threads].set(dct[py], row[py], ctab, size.x, inverse);
			}

			for (i = 0; i < threads; i++) {
				pthread_create(&pid[i], NULL, &DCTqueue_run, queue[i]);
			}

			for (i = 0; i < threads; i++) {
				void *ret;

				pthread_join(pid[i], &ret);
			}

			for (py = 0; py < size.y; py++) {
				if (!inverse) {
					p[(py * size.x)].color[0] = dct[py][0];
					for (px = 1; px < size.x; px++) {
						float m = (1 + (((float)px / (float)size.x) * str));
						if (m <= 0) m = 0;
						p[(py * size.x) + px].color[0] = dct[py][px] * m;
					}
				} else {
					for (px = 0; px < size.x; px++) {
						p[(py * size.x) + px].color[0] = dct[py][px] * (str + 1.0);
					}
				}
			}
			
			clock_gettime(CLOCK_MONOTONIC, &t2);
			etime = (((double)t2.tv_sec - (double)t1.tv_sec) * 1000000000) + (double)t2.tv_nsec - (double)t1.tv_nsec;
			iter = (((double)size.x * (double)size.x) * (double)size.y * 2.0);
			fprintf(stderr, "time:%lf nsec flops:%lf gflops\n", etime, iter / etime);
			free(dct); free(row); free(ctab); delete(queue);

			return;

		}

		void Darken(T bright) {
			int i;
			T bottom = 5.0, peak = -5.0, m, mpeak = -5.0, tot = 0.0, mult;

			for (i = 0; i < size.a(); i++) {
				if (p[i].rgb.y < bottom) bottom = p[i].rgb.y;
				if (p[i].rgb.y > peak) peak = p[i].rgb.y;
			}

			fprintf(stderr, "%lf %lf ", bottom, peak);

			for (i = 0; i < size.a(); i++) {
				tot += p[i].rgb.y;
	
				if ((m = (p[i].rgb.y * p[i].rgb.r)) > peak) mpeak = m;
				if ((m = (p[i].rgb.y * p[i].rgb.g)) > peak) mpeak = m;
				if ((m = (p[i].rgb.y * p[i].rgb.b)) > peak) mpeak = m;
			}
			fprintf(stderr, "%lf %lf\n", (tot / size.a()), mpeak);

			tot /= size.a();
			for (i = 0; i < size.a(); i++) {
				p[i].rgb.y /= mpeak;
			}

		}
};


int main(int argc, char *argv[])
{
#if 0
	RGBY<float> c;
	HSL<float> h;

	c.r = 0.5; c.g = 0.8; c.b = 0.8;
	h = h.FromRGB(c);
	fprintf(stderr, "%f %f %f\n", h.h, h.s, h.l);
	c = c.FromHSL(h);

	fprintf(stderr, "%f %f %f\n", c.r, c.g, c.b);




return 0;
#endif
	char line1[512], line2[512], line3[512];
	int i = 0, x, y;
	float p1 = 1.0, p2 = 1.0, p3 = 1.0, p4 = 0.0, p5 = 0.0, avg;
	unsigned short int *ipic, *opic;
	Picture<float> *pic, *ypic, *tmppic, *tmppic2, *rpic1, *rpic2;

	if (argc >= 2) sscanf(argv[1], "%f", &p1);	
	if (argc >= 3) sscanf(argv[2], "%f", &p2);	
	if (argc >= 4) sscanf(argv[3], "%f", &p3);	
	if (argc >= 5) sscanf(argv[4], "%f", &p4);	
	if (argc >= 6) sscanf(argv[5], "%f", &p5);	

	// TODO:  rewrite this into a cleaner form in Picture
	memset(line1, 0, 512);

	/* read the first line */
	while (read(0, &line1[i], 1)) {
		if (line1[i] == '\n') break;
		i++;
	}
	line1[i + 1] = 0;
	
	memset(line2, 0, 512);
	i = 0;
	/* read the second line */
	while (read(0, &line2[i], 1)) {
		if (line2[i] == '\n') break;
		i++;
	}
	line2[i + 1] = 0;

	sscanf(line2, "%d %d", &x, &y);

	memset(line3, 0, 512);
	i = 0;
	/* read the third line */
	while (read(0, &line3[i], 1)) {
		if (line3[i] == '\n') break;
		i++;
	}
	line3[i + 1] = 0;

	ipic = (unsigned short *)malloc(x * y * 6);
	opic = (unsigned short *)malloc(x * y * 6);
	read(0, ipic, (x * y * 6)); 
	for (i = 0; i < (x * y * 3); i++) ipic[i] = ntohs(ipic[i]);

	pic = new Picture<float>(Point(x, y), ipic);
	ypic = new Picture<float>(Point(x, y), NULL);
//	tmppic = new Picture<float>(Point(x, y), NULL);
//	tmppic2 = new Picture<float>(Point(x, y), NULL);

	ypic->RGBtoHSL(*pic);
	avg = ypic->GetAvg();
#if 0
	float set[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	float set2[8] = {4, 0, 0, 0, 0, 0, 0, 0};
	for (i = 0; i < 8; i++) {
		Pixel<float> cur = ypic->Get(i, 0);
		cur.y = set[i];
		ypic->Set(i, 0, cur);

		cur = ypic->Get(i, 1);
		cur.y = set2[i];
		ypic->Set(i, 1, cur);
	}
#endif
	tmppic = new Picture<float>(*ypic);
	ypic->thDCT(*ypic, false, 0, p1);
	tmppic2 = new Picture<float>(*ypic);
	rpic1 = new Picture<float>(*ypic, 1);
	rpic1->thDCT(*rpic1, false, 0, p2);
	rpic1->thDCT(*rpic1, true, 0, 0.0);
	rpic2 = new Picture<float>(*rpic1, 1);
	rpic2->thDCT(*rpic2, true, 0, 0.0);

	if (0) {
		ypic->thDCT(*rpic2, false, 2, p1);
		tmppic2 = new Picture<float>(*ypic);
		rpic1 = new Picture<float>(*ypic, 1);
		rpic1->thDCT(*rpic1, false, 2, p2);
		rpic1->thDCT(*rpic1, true, 2, 0.0);
		rpic2 = new Picture<float>(*rpic1, 1);
		rpic2->thDCT(*rpic2, true, 2, 0.0);		
	}

#if 0
	for (iy = 0; iy < 2; iy++) {
		for (ix = 0; ix < 8; ix++) {
			fprintf(stderr, "%d,%d: %lf %lf %lf %lf\n", ix, iy, tmppic->Get(ix,iy).rgb.y, ypic->Get(ix,iy).rgb.y, rpic2->Get(ix, iy).rgb.y, rpic2->Get(ix, iy).rgb.y);
		}
	}
#endif

//	rpic2->Darken(1.0);
	rpic2->Boost(*tmppic, *rpic2, p3, p4, p5);
	pic->HSLtoRGB(*rpic2);
//	pic->Normalize(p3);

	pic->Export(opic);

	for (i = 0; i < (x * y * 3); i++) opic[i] = htons(opic[i]);
		
	write(1, line1, strlen(line1));
	write(1, line2, strlen(line2));
	write(1, line3, strlen(line3));
	write(1, opic, x * y * 6);
	return 0;
}
