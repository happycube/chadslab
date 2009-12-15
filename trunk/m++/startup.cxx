char stack[1024];

typedef unsigned char uint8;
typedef unsigned char byte;
typedef unsigned long uint32;
typedef long int32;

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

class MObject {
	public:
		MObject() {};
};

class MDevice : public MObject {
	protected:
		/* Most devices have a register block */
		byte *regbase;
		int   regsize;
	public:
		MDevice() : MObject() {};
};

class M_UART : public MDevice {
	public:
		M_UART() : MDevice() {};
		virtual uint32 setBaud(uint32 baud) = 0;
		virtual bool rxPoll() = 0;
		virtual int32 Read() = 0;
		virtual bool  Write(byte val) = 0;
};

/* Registers for the Stellaris(r) UART */

/* Register names and fields are directly from the LM3S811 data sheet */

#define MUS_UARTDR   0x00

#define MUS_UARTFR   0x18 >> 2
#define MUS_UARTFR_TXFE 0x80
#define MUS_UARTFR_RXFF 0x40
#define MUS_UARTFR_TXFF 0x20
#define MUS_UARTFR_RXFE 0x10
#define MUS_UARTFR_BUSY 0x08

#define MUS_UARTIBRD 0x24 >> 2
#define MUS_UARTFBRD 0x28 >> 2

#define MUS_UARTCRH 0x2C >> 2
#define MUS_UARTCRH_FEN  0x10
#define MUS_UARTCRH_WLEN 0x20 // actually 2 bits

#define MUS_UARTCTL 0x30 >> 2

/*
 * XXX:
 * 
 * - Very skeletal.
 * - No interrupt support yet.
 * - Not even tested on real hardware.
 */

uint32 *ptr;

class M_UART_Stellaris : public M_UART {
	protected:
		uint32 *regbase32;
		uint32 uart_sysclock;
		bool setEnable(bool en) {
			regbase32[MUS_UARTCTL] = (regbase32[MUS_UARTCTL] & ~0x01) | (en ? 0x01 : 0x00);
			if (!en) {
				/* XXX : wait for everything to stop */
				regbase32[MUS_UARTCRH] &= ~MUS_UARTCRH_FEN;
			}	
			return true;
		};
	public:
		M_UART_Stellaris(uint32 clock = 2*(10^7), byte *rb = (byte *)0x4000c000) : M_UART() {
			uart_sysclock = clock;
			regbase = rb; 
			regbase32 = (uint32 *)regbase; 
			regbase32[MUS_UARTCRH] = 0x60; // N,8,1
			setBaud(115200);
		};
		virtual uint32 setBaud(uint32 baud = 115200) {
			uint32 baud16, frac;

			setEnable(false);
	
			/* This is based on section 12.3 of the LM3S811 DS,
			 * converted into 32-bit integer math */	
			baud16 = (16 * baud);

			regbase32[MUS_UARTIBRD] = (uart_sysclock / baud16);	
			frac = (((uart_sysclock % baud16) * 64) + (baud16 / 2)) / baud16; 
			regbase32[MUS_UARTFBRD] = (frac > 63) ? 63 : frac;
			setEnable(true);
			return baud;
		}
		virtual bool rxPoll() {
			return regbase32[MUS_UARTFR] && MUS_UARTFR_RXFE;	
		}
		virtual int32 Read() {
			/* TODO:  Error handling */
			return (M_UART_Stellaris::rxPoll() ? (regbase32[MUS_UARTDR] & 0xff) : -1);
		}
		virtual bool Write(byte val) {
		//	if (regbase32[MUS_UARTFR] && MUS_UARTFR_TXFF)
		//		return false;

			regbase32[MUS_UARTDR] = (uint32)val;
			return true;
		}
};

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

