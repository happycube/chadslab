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
 	if (addr == 0xd013) {
		printf("%c", b);
	} else 
		mem[addr] = b;
}

void main()
{
	int fd = open("replica13a.bin", O_RDONLY);
	read(fd, &mem[0xe000], 0x2000);
	pc = mread(0xfffc) + (mread(0xfffd) << 8);
	trace = 0;
	while (1) {
		if (1 || trace) printf("a %02x x %02x y %02x flags %02x pc %04x:%02x %02x %02x\n", a, x, y, flags, pc, mem[pc], mem[pc + 1], mem[pc + 2]);
		step();
	}
}
