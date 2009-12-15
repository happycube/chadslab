MEMORY
{
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 0x00010000
    SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 0x00002000
}
SECTIONS {
        .text : {
                 * (.vectors);
                 * (.text);
                 * (.text.*);
                 etext = .;
        } > FLASH
        .data : {
        	sdata = .;
                 * (.data);
                 * (.init_array);
		edata = .;
        } > SRAM AT> FLASH
        __exidx_start = .; 
	.ARM.exidx :
	{
		*(.ARM.exidx* .gnu.linkonce.armexidx.*)
	} >FLASH
	__exidx_end = .; 

	.bss : {
                 sbss = .;
                 * (.bss);
                 ebss = .;
        } > SRAM
        .rodata : {
                 * (.rodata.*);
        } > FLASH
}

