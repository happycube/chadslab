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
#include <pthread.h>

typedef struct HSL_t {
	float h, s, l;
} HSL;

typedef struct RGB_t {
	float r, g, b;
} RGB;

inline float clip(float in) 
{
	if (in < 0.0) in = 0.0;
	else if (in > 1.0) in = 1.0;
	
	return in;
}

inline RGB hsltorgb(HSL in) {
	RGB out;
	float t[3], q, p;
	int i;

	in.l = clip(in.l);

	if (in.s < 0.0) {
		out.r = out.g = out.b = in.l;
		return out;
	}

	t[1] = fmod(in.h, 1.0); // g
	t[0] = t[1] + (1.0/3.0); // r
	t[2] = t[1] - (1.0/3.0); // b
//	fprintf(stderr, "t %2f,%2f,%2f\n", t[0], t[1], t[2]);

	if (in.l < 0.5)
		q = in.l * (1.0 + in.s);
	else
		q = (in.l + in.s) - (in.l * in.s);

	p = (2 * in.l) - q;
	for (i = 0; i <= 2; i++) {
		while (t[i] < 0.0) t[i] += 1.0;
		while (t[i] > 1.0) t[i] -= 1.0;

		if ((t[i] * 6) < 1.0) t[i] = p + (q - p) * 6.0 * t[i];
		  else if (t[i] < 0.5) t[i] = q;
		  else if ((3 * t[i]) < 2) t[i] = p + (q - p) * (6.0 * ((2.0 / 3.0) - t[i]));
		  else t[i] = p;
	}
	out.r = t[0]; out.g = t[1]; out.b = t[2];

//	fprintf(stderr, "-> %2f,%2f,%2f\n", out.r, out.g, out.b);

	return out;
}	

inline HSL rgbtohsl(RGB old) {
	HSL c;
	float min, max, hbase;
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

//	fprintf(stderr, "%lf %lf ", min, max);

	c.s = (max - min) / ((c.l > 0.5) ? (2 - (2 * c.l)) : (2 * c.l));

	if (colmax == 1) hbase = old.b - old.r;
	else if (colmax == 2) hbase = old.r - old.g;
	else if (colmax == 3) hbase = old.g - old.b;
		
	c.h = ((60.0 * hbase) / (max - min)) + (colmax * 120.0);
//	fprintf(stderr, "%f %f %f %d %f\n", old.r, old.g, old.b, colmax, hbase);
	c.h = fmodf(c.h, 360.0) / 360.1;

	//fprintf(stderr, "-> %2f,%2f,%2f\n", c.h, c.s, c.l);

	return c;
}

int main(int argc, char *argv[])
{
	RGB r, o;
	HSL h;
	char line1[512], line2[512], line3[512];
	int i = 0, x, y;
	float p1 = 1.0, p2 = 1.0, p3 = 1.0, p4 = 0.0, p5 = 0.0, avg;
	unsigned short int *ipic, *opic;
	float *_rpic, *_hpic;
	RGB *rpic;
	HSL *hpic;
	float hil = 0.0, his = 0.0;
	int lgram[1000];
	int sgram[1000];
	
	memset(lgram, 0, sizeof(lgram));
	memset(sgram, 0, sizeof(sgram));
	
	if (argc >= 2) sscanf(argv[1], "%f", &p1);	
	if (argc >= 3) sscanf(argv[2], "%f", &p2);	

	// TODO:  rewrite this
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
	_rpic = (float *)malloc(x * y * 12);
	_hpic = (float *)malloc(x * y * 12);
	read(0, ipic, (x * y * 6)); 
	
	rpic = (RGB *)_rpic;
	hpic = (HSL *)_hpic;
#ifdef RGB	
	for (i = 0; i < (x * y * 3); i++) ipic[i] = ntohs(ipic[i]);
	for (i = 0; i < (x * y * 3); i++) _hpic[i] = ((float)ipic[i]) / 65536.0;

	for (i = 0; i < (x * y); i++) rpic[i] = hsltorgb(hpic[i]);

	for (i = 0; i < (x * y * 3); i++) opic[i] = htons((unsigned short int)(_rpic[i] * 65536.0));
#else
	for (i = 0; i < (x * y * 3); i++) ipic[i] = ntohs(ipic[i]);
	for (i = 0; i < (x * y * 3); i++) _rpic[i] = ((float)ipic[i]) / 65536.0;

	// convert pictures from RGB to HSL and take a histogram of L and S for later
	for (i = 0; i < (x * y); i++) {
//	    fprintf(stderr, "%4.4f %4.4f %4.4f -> ", rpic[i].r, rpic[i].g, rpic[i].b);
	    hpic[i] = rgbtohsl(rpic[i]);
//	    fprintf(stderr, "%4.4f %4.4f %4.4f -> ", hpic[i].h, hpic[i].s, hpic[i].l);
//	    rpic[i] = hsltorgb(hpic[i]);
//	    fprintf(stderr, "%4.4f %4.4f %4.4f\n", rpic[i].r, rpic[i].g, rpic[i].b);
	    lgram[(int)(hpic[i].l * 999)]++;
	    sgram[(int)(hpic[i].s * 999)]++;
	}
	
	// based on command line parameters, determine mult factors for L and S
	// (params are desired % of pixels <= a given L or S value to use as mult)
	int j = 0, tot = 0;
	int tgt = (int)((float)(x * y) * p1);
	for (i = 0; i < 1000 && (tot < tgt); i++, tot += lgram[i]); 
	float lmul = 1000.0 / (float)i;
	
	tgt = (int)((float)(x * y) * p2);
	for (i = 0, tot = 0; i < 1000 && (tot < tgt); i++, tot += sgram[i]);
	float smul = 1000.0 / (float)i;

	for (i = 0; i < 1000; i++) fprintf(stderr, "%d %d\n", lgram[i], sgram[i]);
	
	fprintf(stderr, "%d %f %f\n", i, lmul, smul);

	for (i = 0; i < (x * y); i++) hpic[i].l = fmin(hpic[i].l * lmul, 1.0);
	for (i = 0; i < (x * y); i++) hpic[i].s = fmin(hpic[i].s * smul, 1.0);

	for (i = 0; i < (x * y); i++) rpic[i] = hsltorgb(hpic[i]);
	for (i = 0; i < (x * y * 3); i++) opic[i] = htons((unsigned short int)(_rpic[i] * 65535.0));
	
//	for (i = 0; i < (x * y * 3); i++) opic[i] = htons((unsigned short int)(_hpic[i] * 65535.0));
#endif
	write(1, line1, strlen(line1));
	write(1, line2, strlen(line2));
	write(1, line3, strlen(line3));
	write(1, opic, x * y * 6);

	return 0;
}