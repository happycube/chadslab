#include "mpp.h"
#include "l3s_serial.h"

char stack[1024];

extern char sdata;
extern char edata;
extern char etext;
extern char sbss;
extern char ebss;
static char *bssp = (char *)0xDEADBEEF;

extern "C" void _GLOBAL__I_stack();

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
	_GLOBAL__I_stack();
	main();
   
	while (1);
}

void nmi_isr()
{
}

void hardfault_isr()
{
}

void svcall_isr()
{
}

/* These interrupt vectors are forced to 0x00000000 by the linker script */

__attribute__ ((section(".vectors")))
void *vectors[] = {
        stack + sizeof(stack), /* initial stack pointer */
        (void *)start,
        (void *)nmi_isr,
        (void *)hardfault_isr,
        0, /* mpu_isr */
        0, /* busfault_isr */
        0, /* usagefault_isr */
        0, 0, 0, 0, /* reserved vectors */
        (void *)svcall_isr, /*  */
        0, /* debugmonitor_isr */
        0, /* reserved vector */
        0, /* pendable_service_isr */
        0, /* systick_service_isr */
        0, /* external-1 - there can be 15-16 of these?*/
};

/* from http://zedcode.blogspot.com/2007/02/gcc-c-link-problems-on-small-embedded.html */
extern "C" void __cxa_pure_virtual(void)
{
// call to a pure virtual function happened ... wow, should never happen ... stop
while(1)
;
}
#if 0
extern "C" void __aeabi_unwind_cpp_pr1()
{
}
#endif
extern "C" void abort() {while(1);}

extern "C" void memcpy(byte *to, byte *from, int len)
{
	for (int i = 0; i < len; i++) *to++ = *from++; 
}


M_UART_Stellaris uart;

char txt[] = "Hello world!\n";

int num1 = 0x40;
int num2 = 0x50;
int result = 0x0;
int main()
{
	char *ptxt = txt;

	while (*ptxt) {	
		while (!uart.Write(*ptxt));
		ptxt++;
	}
        result = num1 + num2;
        return 0;
}

