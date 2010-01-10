#include "mpp.h"
#include "lm3.h"

char stack[1024];

extern char sdata;
extern char edata;
extern char etext;
extern char sbss;
extern char ebss;
static char *bssp = (char *)0xDEADBEEF;

int main();

void start()
{
        char *from, *to;
        /* Initialize .data */
        from = &etext;
        to = &sdata;
        while (to != &edata) {
                 *to++ = *from++;
        }

        /* Clear .bss */
        bssp = &sbss;
        while (bssp != &ebss) {
                 *bssp++ = 0;
        }
//	_GLOBAL__I_stack();
	main();
   
	while (1);
}

void __attribute__((__interrupt__)) nmi_isr()
{
}

void __attribute__((__interrupt__)) hardfault_isr()
{
}

void __attribute__((__interrupt__)) svcall_isr()
{
}

void __attribute__((__interrupt__)) busfault_isr()
{
while (1);
}

void __attribute__((__interrupt__)) usagefault_isr()
{
while (1);
}

/* These interrupt vectors are forced to 0x00000000 by the linker script */

__attribute__ ((section(".vectors")))
void *vectors[] = {
        stack + sizeof(stack), /* initial stack pointer */
        (void *)start,
        (void *)nmi_isr,
        (void *)hardfault_isr,
        0, /* mpu_isr */
        (void *)busfault_isr, /* busfault_isr */
        (void *)usagefault_isr, /* usagefault_isr */
        0, 0, 0, 0, /* reserved vectors */
        (void *)svcall_isr, /*  */
        0, /* debugmonitor_isr */
        0, /* reserved vector */
        0, /* pendable_service_isr */
        0, /* systick_service_isr */
        0, /* external-1 - there can be 15-16 of these?*/
};

void abort() {while(1);}
/*
void memcpy(byte *to, byte *from, int len)
{
	for (int i = 0; i < len; i++) *to++ = *from++; 
}
*/
char txt[] = "Hello world!\n\r";
byte crlf[] = "\n\r";

void writenum(uint32 i)
{
	uint32 j = 1000000000;
	byte c = '0';

	if (i == 0) {
		uart_write(0, &c, 1);
	} else {
		while (i < j) j /= 10; 
		do {
			c = '0' + ((i / j) % 10);
			uart_write(0, &c, 1);
			j /= 10;
		} while (j);
	}
	uart_write(0, crlf, 2);
}

int main()
{
	byte *ptxt = (byte *)txt;
	byte b;
	uint32 p, i;

	platform_init();
	uart_setbaud(0, 115200);

	while (*ptxt) {	
		ptxt += uart_write(0, ptxt, 1);
	}
#if 0
	writenum(1);
	writenum(2);
	for (p = 5; p < (2<<30); p += 2) {
		for (i = 3; ((i * 2) < p) && (p % i); i+= 2);
		if (p % i) writenum(p);
	} 
#endif
	while(1) {
		if (uart_read(0, &b, 1) == 1) uart_write(0, &b, 1);
	}
        return 0;
}


