/* basic macros for Luminary chips, specifically for now the 9B92 */

/*
 * These are intentionally quirky so that they're different from Luminary's header file,
 * which has license restrictions that render them non-free.
 */

/* Clock registers */
#define LM3_SYS_RCC1 0x400fe060
#define LM3_SYS_RCC2 0x400fe070

/* XXX define RCC1 and RCC2 bits here */

#define LM3_SYS_RIS 0x400fe050
#define LM3_SYS_RIS_PLLLOCK 0x40

/* These are the magic values to start the 9B92's clock and get it to 50mhz.
 * The # in the define is the RCC to write to */

/* These are built around a 16mhz external osc right now */
#define LM3_9B92_STARTOSC1 0x078e3d40

/* wait a lil', then... */
#define LM3_9B92_SETOSC2   0x078e3d40

#define LM3_9B92_STARTPLL1 0x078e1d40 
#define LM3_9B92_STARTPLL2 0x07c04800

/* wait for PLL lock then... */

#define LM3_9B92_SETPLL1 0x01841540 // no write to RCC2 needed.

/* Clock gating registers */

/* set the two below to 0x00000001 to get UART0 going */
#define LM3_SYS_RCGC1 0x400fe104
#define LM3_SYS_RCGC2 0x400fe108


