
/*

Copyright (C) 2009-2010 Chad Page. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include "m6502.h"

u8 mem[65540];

inline unsigned char mread(unsigned short addr)
{
	if (trace) printf("read %04x %02x\n", addr, mem[addr]);
        return mem[addr];
}

inline void mwrite(u16 addr, u8 b)
{
	if (trace) printf("write %04x %02x\n", addr, b);
 	if (addr == 0xd012) {
		printf("%c", b);
	} else 
		mem[addr] = b;
}
#if 0
void main()
{
//	FILE *f = fopen("TTL6502.BIN", "r");
	int fd = open("TTL6502.BIN", O_RDONLY);
	int rv, i;

	memset(mem, 0, 65536);

/*
	do {
		unsigned short addr;
		unsigned char c[8];

		// rv = fscanf(f, "%4.4x: %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x", &addr, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]);
		rv = fscanf(f, "%4hx: %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx", &addr, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]);

		for (i = 0; i < rv - 1; i++) {
			mem[addr + i] = c[i];
		} 
	} while (rv > 0);
*/
	read(fd, &mem[0xe000], 0x2000);
	pc = mread(0xfffc) + (mread(0xfffd) << 8);
	trace = 1;
	while (1) {
		if (1 || trace) printf("a %02x x %02x y %02x flags %02x sp %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, sp, pc, mem[pc], mem[pc + 1], mem[pc + 2]);
		step();
	}
}

#else
void main()
{
	FILE *f = fopen("wozmon.txt", "r");
	int rv, i;

	memset(mem, 0, 65536);

	do {
		unsigned short addr;
		unsigned char c[8];

		// rv = fscanf(f, "%4.4x: %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x %2.2x", &addr, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]);
		rv = fscanf(f, "%4hx: %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx %2hhx", &addr, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]);

		for (i = 0; i < rv - 1; i++) {
			mem[addr + i] = c[i];
		} 
	} while (rv > 0);
//	read(fd, &mem[0xff00], 0x2000);
	pc = mread(0xfffc) + (mread(0xfffd) << 8);
	trace = 0;
	while (1) {
//		if (1 || trace) printf("a %02x x %02x y %02x flags %02x sp %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, sp, pc, mem[pc], mem[pc + 1], mem[pc + 2]);
		step();
	}
}
#endif
