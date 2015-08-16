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
	int f_offset;
	int f_strlen;		// Longest observed string for the field (rounded up?)
	Type f_type;	
};

long DataStart = 0;
long RecordLen = 0; // part of format 
int NFields = 0;
FieldInfo Field[MAXKEYS];

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

				if (Field[cfield].f_strlen < strlength) {
					//fprintf(stderr, "%d %d %d\n", cfield, Field[cfield].f_strlen, strlength);
					Field[cfield].f_strlen = strlength;
				}
			}
		}

		if (!inQuote && ((data[i] == ',') || (i == (len - 1)))) {
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

//				if (lines >= 1000) return lines;
			} else {
				num = PreProcessLabelLine(&data[begin], i - begin);
			}
	
			begin = i;
		}
	}
	return lines;
}

long ComputeLengths(int lines)
{
	long len = 0;

	int offset = 0;

	for (int cfield = 0; cfield < NFields; cfield++) {
		switch (Field[cfield].f_type) {
			case T_INT:  offset += sizeof(long); break;
			case T_FLOAT:  offset += sizeof(float); break;
			case T_STRING:  {
				Field[cfield].f_strlen = ((Field[cfield].f_strlen / 8) + 1) * 8;
				offset += Field[cfield].f_strlen;
				break;
			}
			default: break;
		};
	}
	RecordLen = offset;
	RecordLen = ((RecordLen / 32) + 1) * 32;

	len += sizeof(RecordLen);
	len += sizeof(unsigned long); // NFieldsOut
	len += sizeof(unsigned long); // beginning of actual data
	len += NFields * sizeof(struct FieldInfo);

	// round data start to 4K	
	len = ((len / 4096) + 1) * 4096;
	DataStart = len;

	len += lines * RecordLen;

	return len;
}

// Pass 2:  Write out everything

void WriteHeader(unsigned char *map, long &offset)
{
	unsigned long NFieldsOut = NFields;

	*((long *)&map[offset]) = RecordLen;
	offset += sizeof(RecordLen);

	*((long *)&map[offset]) = NFieldsOut;
	offset += sizeof(NFieldsOut);
	
	*((long *)&map[offset]) = DataStart;
	offset += sizeof(DataStart);

	memcpy(&map[offset], Field, NFields * sizeof(struct FieldInfo));
	offset += (sizeof(FieldInfo) * NFields);

	offset = DataStart;
}

void WriteLine(unsigned char *map, long &offset, unsigned char *data, int len)
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

		if (!inQuote && (data[i] == ',')) {
			switch (Field[cfield].f_type) {
				case T_INT:
					intout = atol((const char *)&data[cstart]); 
					*((long *)&map[offset]) = intout;
					offset += sizeof(long);
					break; 
				case T_FLOAT:
					floatout = (float)atof((const char *)&data[cstart]); 
					*((float *)&map[offset]) = floatout;
					offset += sizeof(floatout);
					break; 
				case T_STRING:
					memcpy(&map[offset], &data[cstart + 1], (i - cstart - 2));
					offset += Field[cfield].f_strlen; 
			};
			
			cstart = i + 1;
			cfield++;
		}
	} 

	intout = 0;
	floatout = 0;
	memset(strout, 0, sizeof(strout));
	for (; cfield < NFields; cfield++) {
		switch (Field[cfield].f_type) {
			case T_INT:
				*((long *)&map[offset]) = intout;
				offset += sizeof(long);
				break; 
			case T_FLOAT:
				*((float *)&map[offset]) = floatout;
				offset += sizeof(float);
				break; 
			case T_STRING:
				memset(&map[offset], 0, Field[cfield].f_strlen);
				offset += Field[cfield].f_strlen; 
				break; 
		};
	}
}

void Process(unsigned char *map, long &offset, unsigned char *data, int len)
{
	int begin = 0;
	int linecount = 0;

	WriteHeader(map, offset);

	for (int i = 0; i < len; i++) {
		if (data[i] == '\n') {
			if (begin != 0) { 
				WriteLine(map, offset, &data[begin], i - begin);
				linecount++;
				if (!(linecount % 1000)) {
					fprintf(stderr, "Line %d %ld\n", linecount, offset);
					msync(map, (offset / 4096) * 4096, MS_ASYNC);
				}
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
	}

	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		char outfile[256];
		long offset = 0;

		sprintf(outfile, "%s.out", argv[i]);
	
		long len = ComputeLengths(nrec[i]);

		int out_fd = open(outfile, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IROTH);
		if (out_fd < 0) {
			perror(outfile); 
			exit(-1);
		}
		
		ftruncate(out_fd, len);
	
		void *outmap = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
		if (outmap == MAP_FAILED) return -2;

		fprintf(stderr, "Processing %s %ld\n", outfile, len);

		Process((unsigned char *)outmap, offset, (unsigned char *)map[i], st[i].st_size);
		
		msync(outmap, len, MS_SYNC);
		munmap(outmap, len);
		close(out_fd);
	}
	
	for (int i = 1; (i < argc) && (i <= MAX_FILES); i++) {
		munmap(map[i], st[i].st_size);
		close(fd[i]);
	}

	return 0; 
}
