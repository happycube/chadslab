#include "mpp.h"
#include "lm3.h"
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
//	_GLOBAL__I_stack();
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

void busfault_isr()
{
while (1);
}

void usagefault_isr()
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
//	M_UART_Stellaris uart;

	byte *ptxt = (byte *)txt;
	volatile register unsigned long x;

	w(LM3_SYS_RCC1, LM3_9B92_STARTOSC1);
	/* need osc to settle in */
	for (x = 0; x < 10000; x++);
	w(LM3_SYS_RCC2, LM3_9B92_SETOSC2);

	/* power up pll */
	w(LM3_SYS_RCC1, LM3_9B92_STARTPLL1);
	w(LM3_SYS_RCC2, LM3_9B92_STARTPLL2);

	while (!(r(LM3_SYS_RIS) & LM3_SYS_RIS_PLLLOCK));
	/* pll is ready, go to 50mhz */
	w(LM3_SYS_RCC1, LM3_9B92_SETPLL1);
	/* 100mhz oc */
//	w(LM3_SYS_RCC1, 0x00841540);

	w(LM3_SYS_RCGC1, 0x00000001);
	w(LM3_SYS_RCGC2, 0x00000001);

	uart.setBaud(115200);

	while (*ptxt) {	
		ptxt += uart.write(ptxt, 1);
	}
	while (1) {
		if (uart.read(ptxt, 1)) {
			uart.write(ptxt, 1);
			uart.write(ptxt, 1);
		}
	}
        result = num1 + num2;
        return 0;
}


