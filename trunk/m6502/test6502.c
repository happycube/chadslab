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
		if (1 || trace) printf("a %02x x %02x y %02x flags %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, pc, mem[pc], mem[pc + 1], mem[pc + 2]);
		step();
	}
}
