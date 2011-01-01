
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
#include "wozmon.h"
#include "a1basic.h"

u8 mem[4096];
u8 c;

inline unsigned char mread(unsigned short addr)
{
//	if (trace) printf("read %04x %02x\n", addr, mem[addr]);
	if (!c) {
		if (addr == 0xd010) return 0x80;
		if (addr == 0xd011) return 0x00;
      	} else {
		if (addr == 0xd011) return 0x80;
		if (addr == 0xd010) {
			u8 tchar = c;
			if (tchar == '\n') tchar = '\r';
			c = 0;
			return tchar | 0x80;
		} 
	} 

	if ((addr & 0xff00) == 0xff00) {
		return wozmon[addr & 0xff];
	}
	if ((addr & 0xe000) == 0xe000) {
		return a1basic[addr & 0xfff];
	}
	if (addr < sizeof(mem)) { 
		return mem[addr];
	}
	return 0;
}

inline void mwrite(u16 addr, u8 b)
{
 	if (addr == 0xd012 || addr == 0xd0f2) {
	if (trace) printf("write %04x %02x\n", addr, b);
		unsigned char o = b & 0x7f;
		if (o == '\r') o = '\n';
		write(1, &o, 1);
	} else { 
		if (addr > sizeof(mem)) return;
		mem[addr] = b;
	}
}
#if 0
void main()
{
//	FILE *f = fopen("TTL6502.BIN", "r");
	int fd = open("TTL6502.BIN", O_RDONLY);
	int rv, i;

	memset(mem, 0, 65536);

	read(fd, &mem[0xe000], 0x2000);
	pc = mread(0xfffc) + (mread(0xfffd) << 8);
	trace = 1;
	while (1) {
		printf("a %02x x %02x y %02x flags %02x sp %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, sp, pc, mem[pc], mem[pc + 1], mem[pc + 2]);
		step();
	}
}

#else

char test[] = "10 PRINT I\r15 I = I + 1\r20 GOTO 10\rRUN\r";

void main()
{
	int rv, i = 0;

	fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);

	memset(mem, 0, sizeof(mem));

//	read(fd, &mem[0xff00], 0x2000);
	pc = mread(0xfffc) + (mread(0xfffd) << 8);
//	pc = 0xe000;
	trace = 0;
	while (1) {
		if (trace) printf("a %02x x %02x y %02x flags %02x sp %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, sp, pc, mread(pc), mread(pc+1), mread(pc+2));
		step();

//		if (!c && test[i]) c = test[i++];
		if (!c && (read(0, &c, 1) == 1));
	}
}
#endif
