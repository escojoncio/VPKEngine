# _start(rdi = scratch memory): calls vp_main and returns its value to the host's exit address.
.text
.globl _start
_start:
    sub $8, %rsp
    call vp_main
    add $8, %rsp
    ret
