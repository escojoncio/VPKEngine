# x87 transcendental instructions on ordinary arguments: results to the last bit of the extended format
# (the hardware's own are within one unit of the correctly rounded value, and equal to it for most arguments;
# these are the ones where the two agree).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fninit
    fldl 0(%rsi)
    fsin
    fnstsw 0(%rdi)
    fstpt 16(%rdi)
    fninit
    fldl 8(%rsi)
    fsin
    fnstsw 48(%rdi)
    fstpt 64(%rdi)
    fninit
    fldl 16(%rsi)
    fsin
    fnstsw 96(%rdi)
    fstpt 112(%rdi)
    fninit
    fldl 24(%rsi)
    fsin
    fnstsw 144(%rdi)
    fstpt 160(%rdi)
    fninit
    fldl 32(%rsi)
    fsin
    fnstsw 192(%rdi)
    fstpt 208(%rdi)
    fninit
    fldl 40(%rsi)
    fsin
    fnstsw 240(%rdi)
    fstpt 256(%rdi)
    fninit
    fldl 48(%rsi)
    fsin
    fnstsw 288(%rdi)
    fstpt 304(%rdi)
    fninit
    fldl 56(%rsi)
    fsin
    fnstsw 336(%rdi)
    fstpt 352(%rdi)
    fninit
    fldl 64(%rsi)
    fsin
    fnstsw 384(%rdi)
    fstpt 400(%rdi)
    fninit
    fldl 72(%rsi)
    fsin
    fnstsw 432(%rdi)
    fstpt 448(%rdi)
    fninit
    fldl 80(%rsi)
    fsin
    fnstsw 480(%rdi)
    fstpt 496(%rdi)
    fninit
    fldl 88(%rsi)
    fsin
    fnstsw 528(%rdi)
    fstpt 544(%rdi)
    fninit
    fldl 96(%rsi)
    fsin
    fnstsw 576(%rdi)
    fstpt 592(%rdi)
    fninit
    fldl 104(%rsi)
    fsin
    fnstsw 624(%rdi)
    fstpt 640(%rdi)
    fninit
    fldl 112(%rsi)
    fsin
    fnstsw 672(%rdi)
    fstpt 688(%rdi)
    fninit
    fldl 120(%rsi)
    fsin
    fnstsw 720(%rdi)
    fstpt 736(%rdi)
    fninit
    fldl 128(%rsi)
    fsin
    fnstsw 768(%rdi)
    fstpt 784(%rdi)
    fninit
    fldl 136(%rsi)
    fsin
    fnstsw 816(%rdi)
    fstpt 832(%rdi)
    fninit
    fldl 144(%rsi)
    fsin
    fnstsw 864(%rdi)
    fstpt 880(%rdi)
    fninit
    fldl 152(%rsi)
    fsin
    fnstsw 912(%rdi)
    fstpt 928(%rdi)
    fninit
    fldl 160(%rsi)
    fcos
    fnstsw 960(%rdi)
    fstpt 976(%rdi)
    fninit
    fldl 168(%rsi)
    fcos
    fnstsw 1008(%rdi)
    fstpt 1024(%rdi)
    fninit
    fldl 176(%rsi)
    fcos
    fnstsw 1056(%rdi)
    fstpt 1072(%rdi)
    fninit
    fldl 184(%rsi)
    fcos
    fnstsw 1104(%rdi)
    fstpt 1120(%rdi)
    fninit
    fldl 192(%rsi)
    fcos
    fnstsw 1152(%rdi)
    fstpt 1168(%rdi)
    fninit
    fldl 200(%rsi)
    fcos
    fnstsw 1200(%rdi)
    fstpt 1216(%rdi)
    fninit
    fldl 208(%rsi)
    fcos
    fnstsw 1248(%rdi)
    fstpt 1264(%rdi)
    fninit
    fldl 216(%rsi)
    fcos
    fnstsw 1296(%rdi)
    fstpt 1312(%rdi)
    fninit
    fldl 224(%rsi)
    fcos
    fnstsw 1344(%rdi)
    fstpt 1360(%rdi)
    fninit
    fldl 232(%rsi)
    fcos
    fnstsw 1392(%rdi)
    fstpt 1408(%rdi)
    fninit
    fldl 240(%rsi)
    fcos
    fnstsw 1440(%rdi)
    fstpt 1456(%rdi)
    fninit
    fldl 248(%rsi)
    fcos
    fnstsw 1488(%rdi)
    fstpt 1504(%rdi)
    fninit
    fldl 256(%rsi)
    fcos
    fnstsw 1536(%rdi)
    fstpt 1552(%rdi)
    fninit
    fldl 264(%rsi)
    fcos
    fnstsw 1584(%rdi)
    fstpt 1600(%rdi)
    fninit
    fldl 272(%rsi)
    fcos
    fnstsw 1632(%rdi)
    fstpt 1648(%rdi)
    fninit
    fldl 280(%rsi)
    fcos
    fnstsw 1680(%rdi)
    fstpt 1696(%rdi)
    fninit
    fldl 288(%rsi)
    fcos
    fnstsw 1728(%rdi)
    fstpt 1744(%rdi)
    fninit
    fldl 296(%rsi)
    fcos
    fnstsw 1776(%rdi)
    fstpt 1792(%rdi)
    fninit
    fldl 304(%rsi)
    fcos
    fnstsw 1824(%rdi)
    fstpt 1840(%rdi)
    fninit
    fldl 312(%rsi)
    fcos
    fnstsw 1872(%rdi)
    fstpt 1888(%rdi)
    fninit
    fldl 320(%rsi)
    fptan
    fnstsw 1920(%rdi)
    fstpt 1936(%rdi)
    fstpt 1952(%rdi)
    fninit
    fldl 328(%rsi)
    fptan
    fnstsw 1968(%rdi)
    fstpt 1984(%rdi)
    fstpt 2000(%rdi)
    fninit
    fldl 336(%rsi)
    fptan
    fnstsw 2016(%rdi)
    fstpt 2032(%rdi)
    fstpt 2048(%rdi)
    fninit
    fldl 344(%rsi)
    fptan
    fnstsw 2064(%rdi)
    fstpt 2080(%rdi)
    fstpt 2096(%rdi)
    fninit
    fldl 352(%rsi)
    fptan
    fnstsw 2112(%rdi)
    fstpt 2128(%rdi)
    fstpt 2144(%rdi)
    fninit
    fldl 360(%rsi)
    fptan
    fnstsw 2160(%rdi)
    fstpt 2176(%rdi)
    fstpt 2192(%rdi)
    fninit
    fldl 368(%rsi)
    fptan
    fnstsw 2208(%rdi)
    fstpt 2224(%rdi)
    fstpt 2240(%rdi)
    fninit
    fldl 376(%rsi)
    fptan
    fnstsw 2256(%rdi)
    fstpt 2272(%rdi)
    fstpt 2288(%rdi)
    fninit
    fldl 384(%rsi)
    fptan
    fnstsw 2304(%rdi)
    fstpt 2320(%rdi)
    fstpt 2336(%rdi)
    fninit
    fldl 392(%rsi)
    fptan
    fnstsw 2352(%rdi)
    fstpt 2368(%rdi)
    fstpt 2384(%rdi)
    fninit
    fldl 400(%rsi)
    fsincos
    fnstsw 2400(%rdi)
    fstpt 2416(%rdi)
    fstpt 2432(%rdi)
    fninit
    fldl 408(%rsi)
    fsincos
    fnstsw 2448(%rdi)
    fstpt 2464(%rdi)
    fstpt 2480(%rdi)
    fninit
    fldl 416(%rsi)
    fsincos
    fnstsw 2496(%rdi)
    fstpt 2512(%rdi)
    fstpt 2528(%rdi)
    fninit
    fldl 424(%rsi)
    fsincos
    fnstsw 2544(%rdi)
    fstpt 2560(%rdi)
    fstpt 2576(%rdi)
    fninit
    fldl 432(%rsi)
    fsincos
    fnstsw 2592(%rdi)
    fstpt 2608(%rdi)
    fstpt 2624(%rdi)
    fninit
    fldl 440(%rsi)
    fsincos
    fnstsw 2640(%rdi)
    fstpt 2656(%rdi)
    fstpt 2672(%rdi)
    fninit
    fldl 448(%rsi)
    fsincos
    fnstsw 2688(%rdi)
    fstpt 2704(%rdi)
    fstpt 2720(%rdi)
    fninit
    fldl 456(%rsi)
    fsincos
    fnstsw 2736(%rdi)
    fstpt 2752(%rdi)
    fstpt 2768(%rdi)
    fninit
    fldl 464(%rsi)
    fsincos
    fnstsw 2784(%rdi)
    fstpt 2800(%rdi)
    fstpt 2816(%rdi)
    fninit
    fldl 472(%rsi)
    fsincos
    fnstsw 2832(%rdi)
    fstpt 2848(%rdi)
    fstpt 2864(%rdi)
    fninit
    fldl 480(%rsi)
    fsincos
    fnstsw 2880(%rdi)
    fstpt 2896(%rdi)
    fstpt 2912(%rdi)
    fninit
    fldl 488(%rsi)
    fsincos
    fnstsw 2928(%rdi)
    fstpt 2944(%rdi)
    fstpt 2960(%rdi)
    fninit
    fldl 496(%rsi)
    fldl 504(%rsi)
    fpatan
    fnstsw 2976(%rdi)
    fstpt 2992(%rdi)
    fninit
    fldl 512(%rsi)
    fldl 520(%rsi)
    fpatan
    fnstsw 3024(%rdi)
    fstpt 3040(%rdi)
    fninit
    fldl 528(%rsi)
    fldl 536(%rsi)
    fpatan
    fnstsw 3072(%rdi)
    fstpt 3088(%rdi)
    fninit
    fldl 544(%rsi)
    fldl 552(%rsi)
    fpatan
    fnstsw 3120(%rdi)
    fstpt 3136(%rdi)
    fninit
    fldl 560(%rsi)
    fldl 568(%rsi)
    fpatan
    fnstsw 3168(%rdi)
    fstpt 3184(%rdi)
    fninit
    fldl 576(%rsi)
    fldl 584(%rsi)
    fpatan
    fnstsw 3216(%rdi)
    fstpt 3232(%rdi)
    fninit
    fldl 592(%rsi)
    fldl 600(%rsi)
    fpatan
    fnstsw 3264(%rdi)
    fstpt 3280(%rdi)
    fninit
    fldl 608(%rsi)
    fldl 616(%rsi)
    fpatan
    fnstsw 3312(%rdi)
    fstpt 3328(%rdi)
    fninit
    fldl 624(%rsi)
    f2xm1
    fnstsw 3360(%rdi)
    fstpt 3376(%rdi)
    fninit
    fldl 632(%rsi)
    f2xm1
    fnstsw 3408(%rdi)
    fstpt 3424(%rdi)
    fninit
    fldl 640(%rsi)
    f2xm1
    fnstsw 3456(%rdi)
    fstpt 3472(%rdi)
    fninit
    fldl 648(%rsi)
    f2xm1
    fnstsw 3504(%rdi)
    fstpt 3520(%rdi)
    fninit
    fldl 656(%rsi)
    f2xm1
    fnstsw 3552(%rdi)
    fstpt 3568(%rdi)
    fninit
    fldl 664(%rsi)
    f2xm1
    fnstsw 3600(%rdi)
    fstpt 3616(%rdi)
    fninit
    fldl 672(%rsi)
    f2xm1
    fnstsw 3648(%rdi)
    fstpt 3664(%rdi)
    fninit
    fldl 680(%rsi)
    fldl 688(%rsi)
    fyl2x
    fnstsw 3696(%rdi)
    fstpt 3712(%rdi)
    fninit
    fldl 696(%rsi)
    fldl 704(%rsi)
    fyl2x
    fnstsw 3744(%rdi)
    fstpt 3760(%rdi)
    fninit
    fldl 712(%rsi)
    fldl 720(%rsi)
    fyl2x
    fnstsw 3792(%rdi)
    fstpt 3808(%rdi)
    fninit
    fldl 728(%rsi)
    fldl 736(%rsi)
    fyl2x
    fnstsw 3840(%rdi)
    fstpt 3856(%rdi)
    fninit
    fldl 744(%rsi)
    fldl 752(%rsi)
    fyl2x
    fnstsw 3888(%rdi)
    fstpt 3904(%rdi)
    fninit
    fldl 760(%rsi)
    fldl 768(%rsi)
    fyl2x
    fnstsw 3936(%rdi)
    fstpt 3952(%rdi)
    fninit
    fldl 776(%rsi)
    fldl 784(%rsi)
    fyl2x
    fnstsw 3984(%rdi)
    fstpt 4000(%rdi)
    fninit
    fldl 792(%rsi)
    fldl 800(%rsi)
    fyl2x
    fnstsw 4032(%rdi)
    fstpt 4048(%rdi)
    fninit
    fldl 808(%rsi)
    fldl 816(%rsi)
    fyl2xp1
    fnstsw 4080(%rdi)
    fstpt 4096(%rdi)
    fninit
    fldl 824(%rsi)
    fldl 832(%rsi)
    fyl2xp1
    fnstsw 4128(%rdi)
    fstpt 4144(%rdi)
    fninit
    fldl 840(%rsi)
    fldl 848(%rsi)
    fyl2xp1
    fnstsw 4176(%rdi)
    fstpt 4192(%rdi)
    fninit
    fldl 856(%rsi)
    fldl 864(%rsi)
    fyl2xp1
    fnstsw 4224(%rdi)
    fstpt 4240(%rdi)
    fninit
    ret
.balign 8
k:
    .double 1.0
    .double 0.5
    .double 2.0
    .double 3.0
    .double 10.0
    .double 100.0
    .double 100000.0
    .double 10000000000.0
    .double 0.1
    .double -0.7
    .double 0.3
    .double 1.5
    .double 7.0
    .double 0.001
    .double 12345.678
    .double 0.785398
    .double -2.5
    .double 6.0
    .double 1000000000000000.0
    .double 0.01
    .double 1.0
    .double 0.5
    .double 2.0
    .double 3.0
    .double 10.0
    .double 100.0
    .double 100000.0
    .double 10000000000.0
    .double 0.1
    .double -0.7
    .double 0.3
    .double 1.5
    .double 7.0
    .double 0.001
    .double 12345.678
    .double 0.785398
    .double -2.5
    .double 6.0
    .double 1000000000000000.0
    .double 0.01
    .double 1.0
    .double 0.5
    .double 10.0
    .double 100.0
    .double 100000.0
    .double 10000000000.0
    .double 0.1
    .double -0.7
    .double 0.3
    .double 1.5
    .double 1.0
    .double 0.5
    .double 2.0
    .double 3.0
    .double 10.0
    .double 100.0
    .double 100000.0
    .double 10000000000.0
    .double 0.1
    .double -0.7
    .double 0.3
    .double 1.5
    .double 1
    .double 2
    .double 3
    .double 4
    .double -1
    .double -1
    .double 0.5
    .double 0.25
    .double 5
    .double -2
    .double 1000.0
    .double 7
    .double 2
    .double 0.001
    .double -0.1
    .double 3
    .double 0.5
    .double -0.3
    .double 0.25
    .double 0.9
    .double -0.9
    .double 0.75
    .double -0.125
    .double 3
    .double 10
    .double 1
    .double 7
    .double 2
    .double 0.3
    .double 1.5
    .double 100000.0
    .double -1
    .double 3
    .double 0.5
    .double 100
    .double 10
    .double 1.25
    .double 1
    .double 3
    .double 1
    .double 0.2
    .double 1
    .double 0.01
    .double -1
    .double 0.15
    .double 1.5
    .double -0.1
