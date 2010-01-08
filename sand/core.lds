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
        __exlds_start = .; 
	.ARM.exlds :
	{
		*(.ARM.exlds* .gnu.linkonce.armexlds.*)
	} >FLASH
        __exlds_end = .; 

	.bss : {
                 sbss = .;
                 * (.bss);
                 ebss = .;
        } > SRAM
        .rodata : {
                 * (.rodata.*);
        } > FLASH
        __extab_start = .; 
	.ARM.extab :
	{
		*(.ARM.extab* .gnu.linkonce.armextab.*)
	} >FLASH
        __extab_end = .; 
}

