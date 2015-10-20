/* csv2bin.c
 *
 * Copyright (C) 2015 Chad Page
 * 
 * (Yes, this program is a mess, but once finished it only needs to be run once ;) )
 */

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#include <string.h>

enum Type {
	T_INT = 0,
	T_FLOAT = 1,	// . observed
	T_STRING = 2	// " observed 
};

#define MAXKEYS 4096 // irl about 1400 
#define MAXLABEL 16 // Max seen in springleaf data is 10 

// Part of output format
struct FieldInfo {
	char f_name[MAXLABEL];  
	long f_begin;		// Beginning of data in file 
	long f_length;		// Length of each entry (2,4,8, and ? for string) 
	Type f_type;	
};

long DataStart = 0;
long RecordLen = 0; // part of format 
int NFields = 0;
FieldInfo Field[MAXKEYS];

void *FieldData[MAXKEYS];

long flen = 0;

int maxlen = 500000;

int PreProcessLabelLine(unsigned char *data, int len)
{
	bool inQuote = false;
	int num = 0;
	int cstart = 0;
	int cnum = 0;

	int maxlen = 0;

	// This only needs to be run once - the test set won't have differences, I hope.
	if (NFields) return NFields;

	// Pass 1:  Get # of fields
	for (int i = 0; i < len; i++) {
		if (data[i] == '"') {
			inQuote = !inQuote;
			if (inQuote) {
				Field[num].f_type = T_INT; 
				cnum = 0;
			} else {
				Field[num].f_name[cnum++] = 0; 
			}
		} else if (inQuote) {
			if ((cnum + 1) > MAXLABEL) {
				fprintf(stderr, "Fatal error:  maximum label length %d over built max %d\n", cnum + 1, MAXLABEL);
				exit(-1); 
			}
			Field[num].f_name[cnum++] = data[i]; 
		}

		if ((!inQuote) && ((data[i] == ',') || (i == (len - 1)))) {
			if ((i - cstart) > maxlen) maxlen = i - cstart;
			cstart = i + 1;
			num++;
			if (num > MAXKEYS) { 
				fprintf(stderr, "Fatal error:  maximum key %d over built max %d\n", num, MAXKEYS);
				exit(-1); 
			}
		}
	}

	NFields = num;

	return num;
}

int PreProcessLine(unsigned char *data, int len)
{
	bool inQuote = false;
	int cfield = 0;
	int cstart = 0;

	int strstart = -1;

	for (int i = 0; i < len; i++) {
		if (data[i] == '"') {
			inQuote = !inQuote;
			if (inQuote) {
				Field[cfield].f_type = T_STRING;
				strstart = i;
			} else {
				int strlength = (i - strstart) + 1;

				if (Field[cfield].f_length < strlength) {
					//fprintf(stderr, "%d %d %d\n", cfield, Field[cfield].f_length, strlength);
					Field[cfield].f_length = strlength;
				}
			}
		}

		if (!inQuote && ((data[i] == ',') || (i == (len - 1)))) {
			if (Field[cfield].f_type == T_INT) {
				long tmp_abs = abs(atoi((const char *)&data[cstart]));

				if (tmp_abs > Field[cfield].f_length) Field[cfield].f_length = tmp_abs; 
			} 
			cstart = i + 1;
			cfield++;
		}

		if (!inQuote && (data[i] == '.')) {
			if (Field[cfield].f_type == T_INT) Field[cfield].f_type = T_FLOAT;
		}
	} 
	
//	if (dotcount > 0) printf("%d\n", dotcount);
	return cfield;
}

// Pass 1: Determine data types
int PreProcess(unsigned char *data, int len)
{
	int begin = 0;
	int lines = 0;

	for (int i = 0; i < len; i++) {
		if (data[i] == '\n') {
			int num = 0;

			if (begin != 0) { 
				num = PreProcessLine(&data[begin], i - begin);
				lines++;

				if (lines >= maxlen) return lines;
			} else {
				num = PreProcessLabelLine(&data[begin], i - begin);
			}
	
			begin = i;
		}
	}
	return lines;
}

long DetermineLength(int nLines)
{
	int offset = (NFields * sizeof(FieldInfo));

	offset = ((offset / 4096) + 1) * 4096;

	for (int cfield = 0; cfield < NFields; cfield++) {
		switch (Field[cfield].f_type) {
			case T_INT:
				if (Field[cfield].f_length >= INT_MAX) Field[cfield].f_length = 8;
				else if (Field[cfield].f_length > SHRT_MAX) Field[cfield].f_length = 4;
				else Field[cfield].f_length = 2;

//				offset += sizeof(long);
				break;
			case T_FLOAT:  
				Field[cfield].f_length = 4;
				break;
			case T_STRING: 
				Field[cfield].f_length = ((Field[cfield].f_length / 8) + 1) * 8;
				offset += Field[cfield].f_length;
				break;
			default: break;
		};

		Field[cfield].f_begin = offset;
		offset += (Field[cfield].f_length * nLines); 
		offset = ((offset / 256) + 1) * 256;
	}

	fprintf(stderr, "length %d %d\n", nLines, offset);

	return offset;
}

long ComputeLengths(int lines)
{
	long len = 0;

	printf("computelengths %d \n", lines); 

	len = 32 + (NFields * sizeof(struct FieldInfo));

	// round data start to 4K	
	len = ((len / 4096) + 1) * 4096;
	DataStart = len;

//	len += lines * RecordLen;

	return len;
}

// Pass 2:  Write out everything

void WriteHeader(unsigned char *map, long nLines)
{
	unsigned long NFieldsOut = NFields;

	*((long *)&map[0]) = NFieldsOut;
	
	*((long *)&map[8]) = nLines;
	
	memcpy(&map[32], Field, NFields * sizeof(struct FieldInfo));
}

void WriteLine(unsigned char *map, long &record, unsigned char *data, int len)
{
	bool inQuote = false;
	int cfield = 0;
	int cstart = 0;

	long intout;
	float floatout;
	char strout[256];

	for (int i = 0; i < len; i++) {
		if (data[i] == '"') {
			inQuote = !inQuote;
		}

		if (!inQuote && ((data[i] == ',') || (i == (len - 1)))) {
			long offset = Field[cfield].f_begin + (Field[cfield].f_length * record);

			if (offset > flen) {
				fprintf(stderr, "huh %ld %ld %ld\n", offset, record, flen);
			}

			switch (Field[cfield].f_type) {
				case T_INT:
					intout = atol((const char *)&data[cstart]); 

					switch (Field[cfield].f_length) {
					case 2:
						*((short *)&map[offset]) = (short)intout;
						break;
					case 4:
						*((int *)&map[offset]) = (int)intout;
						break;
					case 8:
						*((long *)&map[offset]) = intout;
						break;
					};
					break; 
				case T_FLOAT:
					floatout = (float)atof((const char *)&data[cstart]); 
					*((float *)&map[offset]) = floatout;
					break; 
				case T_STRING: {
					int wlen = i - cstart - 1;
					
					memset(&map[offset], 0, Field[cfield].f_length); 
					if (wlen > (Field[cfield].f_length - 1)) wlen = Field[cfield].f_length - 1;
					if (wlen > 0) {
						memset(&map[offset], 0, Field[cfield].f_length); 
						memcpy(&map[offset], &data[cstart + 1], wlen);
	//					fprintf(stderr, "%s\n", &map[offset]);
					}
				}
					break;
			};
			
			cstart = i + 1;
			cfield++;
		}
	} 

//	fprintf(stderr, "%d\n", cfield);

	intout = -1;
	floatout = -1;
	memset(strout, 0, sizeof(strout));
	for (; cfield < NFields; cfield++) {
		long offset = Field[cfield].f_begin + (Field[cfield].f_length * record);
		fprintf(stderr, "-1'ing %d\n", cfield);
		switch (Field[cfield].f_type) {
			case T_INT:
				switch (Field[cfield].f_length) {
				case 2:
					*((short *)&map[offset]) = (short)intout;
					break;
				case 4:
					*((int *)&map[offset]) = (int)intout;
					break;
				case 8:
					*((long *)&map[offset]) = intout;
					break;
				};
				break; 
			case T_FLOAT:
				*((float *)&map[offset]) = floatout;
				break; 
			case T_STRING:
				memset(&map[offset], 0, Field[cfield].f_length);
				break; 
		};
	}
}

void Process(unsigned char *map, long &record, unsigned char *data, int len)
{
	int begin = 0;
	int linecount = 0;

	for (int i = 0; i < len; i++) {
		if (data[i] == '\n') {
			if (begin != 0) { 
				WriteLine(map, record, &data[begin], i - begin);
				linecount++;
				record++;
				if (!(linecount % 1000)) {
					fprintf(stderr, "Line %d\n", linecount);
//					return;
				}

				if (linecount >= maxlen) return;
			}
	
			begin = i;
		}
	}
}

#define MAX_FILES 4

// Quick and dirty file opening/mapping
int main(int argc, char *argv[])
{
	int fd[MAX_FILES];
	struct stat st[MAX_FILES];
	void *map[MAX_FILES];
	int nrec[MAX_FILES];
	int ntotal = 0;

	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		// fstat will fail with -EBADF if open does
		fd[i] = open(argv[1], O_RDONLY);
		if (fstat(fd[i], &st[i]) == -1) return -1; 	

		map[i] = mmap(NULL, st[i].st_size, PROT_READ, MAP_PRIVATE, fd[i], 0);
		if (map[i] == MAP_FAILED) return -2;

//		PreProcess((unsigned char *)map[i], st[i].st_size);
	}
		
	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		fprintf(stderr, "Pre-processing %s\n", argv[i]);
		nrec[i] = PreProcess((unsigned char *)map[i], st[i].st_size);
		ntotal += nrec[i];
	}

	flen = DetermineLength(ntotal);

	char outfile[256];

	sprintf(outfile, "output-test.bin");

	int out_fd = open(outfile, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IROTH);
	if (out_fd < 0) {
		perror(outfile); 
		exit(-1);
	}
	
	ftruncate(out_fd, flen);

	void *outmap = mmap(NULL, flen, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
	if (outmap == MAP_FAILED) return -2;

	memset(outmap, 0, flen);

	WriteHeader((unsigned char *)outmap, ntotal);
		
	long linesoutput = 0;

	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		Process((unsigned char *)outmap, linesoutput, (unsigned char *)map[i], st[i].st_size);
	}

	munmap(outmap, flen);
	close(out_fd);
	
	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		munmap(map[i], st[i].st_size);
		close(fd[i]);
	}

	return 0; 
}
