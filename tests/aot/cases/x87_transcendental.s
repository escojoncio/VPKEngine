# requires: intel
# (f2xm1 with |x| > 1 and fyl2xp1 with x <= -1 are undefined in the SDM, and AMD's microcode differs from Intel's: the whole file is
# measured on Intel. Elsewhere the translation is compared with the golden file only, which --write-golden never rewrites there.)
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
# Special operands (checked against the host CPU's results): f2xm1 outside [-1, 1] returns the operand
# unchanged (PE); fyl2xp1 with x = -inf is invalid; overflow of the result raises OE, PE (C1 when it rounds
# up to infinity); fpatan(y, +inf) is exactly zero; a tiny, inexact result also raises UE, and one that
# underflows to zero rounds by the mode (RU/RD: the smallest denormal with the sign, C1).
    lea t(%rip), %rdx
    fninit
    fldt 0(%rdx)
    f2xm1
    fnstsw 5120(%rdi)
    fstpt 5136(%rdi)
    fninit
    fldt 16(%rdx)
    f2xm1
    fnstsw 5168(%rdi)
    fstpt 5184(%rdi)
    fninit
    fldt 32(%rdx)
    f2xm1
    fnstsw 5216(%rdi)
    fstpt 5232(%rdi)
    fninit
    fldt 48(%rdx)
    f2xm1
    fnstsw 5264(%rdi)
    fstpt 5280(%rdi)
    fninit
    fldt 64(%rdx)
    f2xm1
    fnstsw 5312(%rdi)
    fstpt 5328(%rdi)
    fninit
    fldt 80(%rdx)
    f2xm1
    fnstsw 5360(%rdi)
    fstpt 5376(%rdi)
    fninit
    fldt 96(%rdx)
    f2xm1
    fnstsw 5408(%rdi)
    fstpt 5424(%rdi)
    fninit
    fldt 112(%rdx)
    f2xm1
    fnstsw 5456(%rdi)
    fstpt 5472(%rdi)
    fninit
    fldt 128(%rdx)
    f2xm1
    fnstsw 5504(%rdi)
    fstpt 5520(%rdi)
    fninit
    fldt 144(%rdx)
    f2xm1
    fnstsw 5552(%rdi)
    fstpt 5568(%rdi)
    fninit
    fldt 160(%rdx)
    f2xm1
    fnstsw 5600(%rdi)
    fstpt 5616(%rdi)
    fninit
    fldt 176(%rdx)
    f2xm1
    fnstsw 5648(%rdi)
    fstpt 5664(%rdi)
    fninit
    fldt 192(%rdx)
    f2xm1
    fnstsw 5696(%rdi)
    fstpt 5712(%rdi)
    fninit
    fldt 208(%rdx)
    f2xm1
    fnstsw 5744(%rdi)
    fstpt 5760(%rdi)
    fninit
    fldt 224(%rdx)
    f2xm1
    fnstsw 5792(%rdi)
    fstpt 5808(%rdi)
    fninit
    fldt 240(%rdx)
    f2xm1
    fnstsw 5840(%rdi)
    fstpt 5856(%rdi)
    fninit
    fldt 256(%rdx)
    f2xm1
    fnstsw 5888(%rdi)
    fstpt 5904(%rdi)
    fninit
    fldt 272(%rdx)
    f2xm1
    fnstsw 5936(%rdi)
    fstpt 5952(%rdi)
    fninit
    fldt 288(%rdx)
    f2xm1
    fnstsw 5984(%rdi)
    fstpt 6000(%rdi)
    fninit
    fldt 304(%rdx)
    f2xm1
    fnstsw 6032(%rdi)
    fstpt 6048(%rdi)
    fninit
    fldt 320(%rdx)
    f2xm1
    fnstsw 6080(%rdi)
    fstpt 6096(%rdi)
    fninit
    fldt 336(%rdx)
    f2xm1
    fnstsw 6128(%rdi)
    fstpt 6144(%rdi)
    fninit
    fldt 352(%rdx)
    f2xm1
    fnstsw 6176(%rdi)
    fstpt 6192(%rdi)
    fninit
    fldt 368(%rdx)
    f2xm1
    fnstsw 6224(%rdi)
    fstpt 6240(%rdi)
    fninit
    fldt 384(%rdx)
    f2xm1
    fnstsw 6272(%rdi)
    fstpt 6288(%rdi)
    fninit
    fldt 400(%rdx)
    f2xm1
    fnstsw 6320(%rdi)
    fstpt 6336(%rdi)
    fninit
    fldt 416(%rdx)
    f2xm1
    fnstsw 6368(%rdi)
    fstpt 6384(%rdi)
    fninit
    fldt 432(%rdx)
    f2xm1
    fnstsw 6416(%rdi)
    fstpt 6432(%rdi)
    fninit
    fldt 448(%rdx)
    f2xm1
    fnstsw 6464(%rdi)
    fstpt 6480(%rdi)
    fninit
    fldt 464(%rdx)
    f2xm1
    fnstsw 6512(%rdi)
    fstpt 6528(%rdi)
    fninit
    fldt 480(%rdx)
    fldt 496(%rdx)
    fyl2xp1
    fnstsw 6560(%rdi)
    fstpt 6576(%rdi)
    fninit
    fldt 512(%rdx)
    fldt 528(%rdx)
    fyl2xp1
    fnstsw 6608(%rdi)
    fstpt 6624(%rdi)
    fninit
    fldt 544(%rdx)
    fldt 560(%rdx)
    fyl2xp1
    fnstsw 6656(%rdi)
    fstpt 6672(%rdi)
    fninit
    fldt 576(%rdx)
    fldt 592(%rdx)
    fyl2xp1
    fnstsw 6704(%rdi)
    fstpt 6720(%rdi)
    fninit
    fldt 608(%rdx)
    fldt 624(%rdx)
    fyl2xp1
    fnstsw 6752(%rdi)
    fstpt 6768(%rdi)
    fninit
    fldt 640(%rdx)
    fldt 656(%rdx)
    fyl2xp1
    fnstsw 6800(%rdi)
    fstpt 6816(%rdi)
    fninit
    fldt 672(%rdx)
    fldt 688(%rdx)
    fyl2xp1
    fnstsw 6848(%rdi)
    fstpt 6864(%rdi)
    fninit
    fldt 704(%rdx)
    fldt 720(%rdx)
    fyl2xp1
    fnstsw 6896(%rdi)
    fstpt 6912(%rdi)
    fninit
    fldt 736(%rdx)
    fldt 752(%rdx)
    fyl2xp1
    fnstsw 6944(%rdi)
    fstpt 6960(%rdi)
    fninit
    fldt 768(%rdx)
    fldt 784(%rdx)
    fyl2xp1
    fnstsw 6992(%rdi)
    fstpt 7008(%rdi)
    fninit
    fldt 800(%rdx)
    fldt 816(%rdx)
    fyl2xp1
    fnstsw 7040(%rdi)
    fstpt 7056(%rdi)
    fninit
    fldt 832(%rdx)
    fldt 848(%rdx)
    fyl2xp1
    fnstsw 7088(%rdi)
    fstpt 7104(%rdi)
    fninit
    fldt 864(%rdx)
    fldt 880(%rdx)
    fyl2x
    fnstsw 7136(%rdi)
    fstpt 7152(%rdi)
    fninit
    fldt 896(%rdx)
    fldt 912(%rdx)
    fyl2x
    fnstsw 7184(%rdi)
    fstpt 7200(%rdi)
    fninit
    fldt 928(%rdx)
    fldt 944(%rdx)
    fyl2x
    fnstsw 7232(%rdi)
    fstpt 7248(%rdi)
    fninit
    fldt 960(%rdx)
    fldt 976(%rdx)
    fyl2x
    fnstsw 7280(%rdi)
    fstpt 7296(%rdi)
    fninit
    fldt 992(%rdx)
    fldt 1008(%rdx)
    fyl2xp1
    fnstsw 7328(%rdi)
    fstpt 7344(%rdi)
    fninit
    fldt 1024(%rdx)
    fldt 1040(%rdx)
    fyl2xp1
    fnstsw 7376(%rdi)
    fstpt 7392(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1056(%rdx)
    fldt 1072(%rdx)
    fyl2x
    fnstsw 7424(%rdi)
    fstpt 7440(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1088(%rdx)
    fldt 1104(%rdx)
    fyl2x
    fnstsw 7472(%rdi)
    fstpt 7488(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1120(%rdx)
    fldt 1136(%rdx)
    fyl2x
    fnstsw 7520(%rdi)
    fstpt 7536(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1152(%rdx)
    fldt 1168(%rdx)
    fyl2x
    fnstsw 7568(%rdi)
    fstpt 7584(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1184(%rdx)
    fldt 1200(%rdx)
    fyl2xp1
    fnstsw 7616(%rdi)
    fstpt 7632(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 1216(%rdx)
    fldt 1232(%rdx)
    fyl2xp1
    fnstsw 7664(%rdi)
    fstpt 7680(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1248(%rdx)
    fldt 1264(%rdx)
    fyl2x
    fnstsw 7712(%rdi)
    fstpt 7728(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1280(%rdx)
    fldt 1296(%rdx)
    fyl2x
    fnstsw 7760(%rdi)
    fstpt 7776(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1312(%rdx)
    fldt 1328(%rdx)
    fyl2x
    fnstsw 7808(%rdi)
    fstpt 7824(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1344(%rdx)
    fldt 1360(%rdx)
    fyl2x
    fnstsw 7856(%rdi)
    fstpt 7872(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1376(%rdx)
    fldt 1392(%rdx)
    fyl2xp1
    fnstsw 7904(%rdi)
    fstpt 7920(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 1408(%rdx)
    fldt 1424(%rdx)
    fyl2xp1
    fnstsw 7952(%rdi)
    fstpt 7968(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1440(%rdx)
    fldt 1456(%rdx)
    fyl2x
    fnstsw 8000(%rdi)
    fstpt 8016(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1472(%rdx)
    fldt 1488(%rdx)
    fyl2x
    fnstsw 8048(%rdi)
    fstpt 8064(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1504(%rdx)
    fldt 1520(%rdx)
    fyl2x
    fnstsw 8096(%rdi)
    fstpt 8112(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1536(%rdx)
    fldt 1552(%rdx)
    fyl2x
    fnstsw 8144(%rdi)
    fstpt 8160(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1568(%rdx)
    fldt 1584(%rdx)
    fyl2xp1
    fnstsw 8192(%rdi)
    fstpt 8208(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 1600(%rdx)
    fldt 1616(%rdx)
    fyl2xp1
    fnstsw 8240(%rdi)
    fstpt 8256(%rdi)
    fninit
    fldt 1632(%rdx)
    fldt 1648(%rdx)
    fpatan
    fnstsw 8288(%rdi)
    fstpt 8304(%rdi)
    fninit
    fldt 1664(%rdx)
    fldt 1680(%rdx)
    fpatan
    fnstsw 8336(%rdi)
    fstpt 8352(%rdi)
    fninit
    fldt 1696(%rdx)
    fldt 1712(%rdx)
    fpatan
    fnstsw 8384(%rdi)
    fstpt 8400(%rdi)
    fninit
    fldt 1728(%rdx)
    fldt 1744(%rdx)
    fpatan
    fnstsw 8432(%rdi)
    fstpt 8448(%rdi)
    fninit
    fldt 1760(%rdx)
    fldt 1776(%rdx)
    fpatan
    fnstsw 8480(%rdi)
    fstpt 8496(%rdi)
    fninit
    fldt 1792(%rdx)
    fldt 1808(%rdx)
    fpatan
    fnstsw 8528(%rdi)
    fstpt 8544(%rdi)
    fninit
    fldt 1824(%rdx)
    fldt 1840(%rdx)
    fpatan
    fnstsw 8576(%rdi)
    fstpt 8592(%rdi)
    fninit
    fldt 1856(%rdx)
    fldt 1872(%rdx)
    fpatan
    fnstsw 8624(%rdi)
    fstpt 8640(%rdi)
    fninit
    fldt 1888(%rdx)
    fldt 1904(%rdx)
    fpatan
    fnstsw 8672(%rdi)
    fstpt 8688(%rdi)
    fninit
    fldt 1920(%rdx)
    fsin
    fnstsw 8720(%rdi)
    fstpt 8736(%rdi)
    fninit
    fldt 1936(%rdx)
    fsin
    fnstsw 8768(%rdi)
    fstpt 8784(%rdi)
    fninit
    fldt 1952(%rdx)
    fsin
    fnstsw 8816(%rdi)
    fstpt 8832(%rdi)
    fninit
    fldt 1968(%rdx)
    fcos
    fnstsw 8864(%rdi)
    fstpt 8880(%rdi)
    fninit
    fldt 1984(%rdx)
    fcos
    fnstsw 8912(%rdi)
    fstpt 8928(%rdi)
    fninit
    fldt 2000(%rdx)
    fcos
    fnstsw 8960(%rdi)
    fstpt 8976(%rdi)
    fninit
    fldt 2016(%rdx)
    fptan
    fnstsw 9008(%rdi)
    fstpt 9024(%rdi)
    fstpt 9040(%rdi)
    fninit
    fldt 2032(%rdx)
    fptan
    fnstsw 9056(%rdi)
    fstpt 9072(%rdi)
    fstpt 9088(%rdi)
    fninit
    fldt 2048(%rdx)
    fptan
    fnstsw 9104(%rdi)
    fstpt 9120(%rdi)
    fstpt 9136(%rdi)
    fninit
    fldt 2064(%rdx)
    fsincos
    fnstsw 9152(%rdi)
    fstpt 9168(%rdi)
    fstpt 9184(%rdi)
    fninit
    fldt 2080(%rdx)
    fsincos
    fnstsw 9200(%rdi)
    fstpt 9216(%rdi)
    fstpt 9232(%rdi)
    fninit
    fldt 2096(%rdx)
    fsincos
    fnstsw 9248(%rdi)
    fstpt 9264(%rdi)
    fstpt 9280(%rdi)
    fninit
    fldt 2112(%rdx)
    fldt 2128(%rdx)
    fpatan
    fnstsw 9296(%rdi)
    fstpt 9312(%rdi)
    fninit
    fldt 2144(%rdx)
    fldt 2160(%rdx)
    fpatan
    fnstsw 9344(%rdi)
    fstpt 9360(%rdi)
    fninit
    fldt 2176(%rdx)
    fldt 2192(%rdx)
    fpatan
    fnstsw 9392(%rdi)
    fstpt 9408(%rdi)
    fninit
    fldt 2208(%rdx)
    fldt 2224(%rdx)
    fpatan
    fnstsw 9440(%rdi)
    fstpt 9456(%rdi)
    fninit
    fldt 2240(%rdx)
    fldt 2256(%rdx)
    fpatan
    fnstsw 9488(%rdi)
    fstpt 9504(%rdi)
    fninit
    fldt 2272(%rdx)
    fldt 2288(%rdx)
    fyl2xp1
    fnstsw 9536(%rdi)
    fstpt 9552(%rdi)
    fninit
    fldt 2304(%rdx)
    fldt 2320(%rdx)
    fyl2xp1
    fnstsw 9584(%rdi)
    fstpt 9600(%rdi)
    fninit
    fldt 2336(%rdx)
    fldt 2352(%rdx)
    fyl2xp1
    fnstsw 9632(%rdi)
    fstpt 9648(%rdi)
    fninit
    fldt 2368(%rdx)
    fldt 2384(%rdx)
    fyl2xp1
    fnstsw 9680(%rdi)
    fstpt 9696(%rdi)
    fninit
    fldt 2400(%rdx)
    fldt 2416(%rdx)
    fyl2xp1
    fnstsw 9728(%rdi)
    fstpt 9744(%rdi)
    fninit
    fldt 2432(%rdx)
    fldt 2448(%rdx)
    fyl2xp1
    fnstsw 9776(%rdi)
    fstpt 9792(%rdi)
    fninit
    fldt 2464(%rdx)
    fldt 2480(%rdx)
    fyl2xp1
    fnstsw 9824(%rdi)
    fstpt 9840(%rdi)
    fninit
    fldt 2496(%rdx)
    fldt 2512(%rdx)
    fyl2x
    fnstsw 9872(%rdi)
    fstpt 9888(%rdi)
    fninit
    fldt 2528(%rdx)
    fldt 2544(%rdx)
    fyl2x
    fnstsw 9920(%rdi)
    fstpt 9936(%rdi)
    fninit
    fldt 2560(%rdx)
    fldt 2576(%rdx)
    fyl2x
    fnstsw 9968(%rdi)
    fstpt 9984(%rdi)
    fninit
    fldt 2592(%rdx)
    fldt 2608(%rdx)
    fyl2x
    fnstsw 10016(%rdi)
    fstpt 10032(%rdi)
    fninit
    fldt 2624(%rdx)
    fldt 2640(%rdx)
    fyl2x
    fnstsw 10064(%rdi)
    fstpt 10080(%rdi)
    fninit
    fldt 2656(%rdx)
    fldt 2672(%rdx)
    fyl2x
    fnstsw 10112(%rdi)
    fstpt 10128(%rdi)
    fninit
    fldt 2688(%rdx)
    fldt 2704(%rdx)
    fyl2x
    fnstsw 10160(%rdi)
    fstpt 10176(%rdi)
    fninit
    fldt 2720(%rdx)
    fldt 2736(%rdx)
    fyl2x
    fnstsw 10208(%rdi)
    fstpt 10224(%rdi)
    fninit
    fldt 2752(%rdx)
    fldt 2768(%rdx)
    fyl2x
    fnstsw 10256(%rdi)
    fstpt 10272(%rdi)
    fninit
    fldt 2784(%rdx)
    fldt 2800(%rdx)
    fyl2x
    fnstsw 10304(%rdi)
    fstpt 10320(%rdi)
    fninit
    fldt 2816(%rdx)
    fldt 2832(%rdx)
    fyl2xp1
    fnstsw 10352(%rdi)
    fstpt 10368(%rdi)
    fninit
    fldt 2848(%rdx)
    fldt 2864(%rdx)
    fyl2xp1
    fnstsw 10400(%rdi)
    fstpt 10416(%rdi)
    fninit
    fldt 2880(%rdx)
    fldt 2896(%rdx)
    fyl2xp1
    fnstsw 10448(%rdi)
    fstpt 10464(%rdi)
    fninit
    fldt 2912(%rdx)
    fldt 2928(%rdx)
    fyl2x
    fnstsw 10496(%rdi)
    fstpt 10512(%rdi)
    fninit
    fldt 2944(%rdx)
    fldt 2960(%rdx)
    fyl2x
    fnstsw 10544(%rdi)
    fstpt 10560(%rdi)
    fninit
    fldt 2976(%rdx)
    fldt 2992(%rdx)
    fyl2xp1
    fnstsw 10592(%rdi)
    fstpt 10608(%rdi)
    fninit
    fldt 3008(%rdx)
    fldt 3024(%rdx)
    fyl2xp1
    fnstsw 10640(%rdi)
    fstpt 10656(%rdi)
    fninit
    fldt 3040(%rdx)
    fldt 3056(%rdx)
    fyl2xp1
    fnstsw 10688(%rdi)
    fstpt 10704(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3072(%rdx)
    fldt 3088(%rdx)
    fyl2x
    fnstsw 10736(%rdi)
    fstpt 10752(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3104(%rdx)
    fldt 3120(%rdx)
    fyl2x
    fnstsw 10784(%rdi)
    fstpt 10800(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3136(%rdx)
    fldt 3152(%rdx)
    fyl2xp1
    fnstsw 10832(%rdi)
    fstpt 10848(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3168(%rdx)
    fldt 3184(%rdx)
    fyl2xp1
    fnstsw 10880(%rdi)
    fstpt 10896(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3200(%rdx)
    fldt 3216(%rdx)
    fyl2xp1
    fnstsw 10928(%rdi)
    fstpt 10944(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3232(%rdx)
    fldt 3248(%rdx)
    fyl2x
    fnstsw 10976(%rdi)
    fstpt 10992(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3264(%rdx)
    fldt 3280(%rdx)
    fyl2x
    fnstsw 11024(%rdi)
    fstpt 11040(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3296(%rdx)
    fldt 3312(%rdx)
    fyl2xp1
    fnstsw 11072(%rdi)
    fstpt 11088(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3328(%rdx)
    fldt 3344(%rdx)
    fyl2xp1
    fnstsw 11120(%rdi)
    fstpt 11136(%rdi)
    fninit
    fldcw cw0f7f(%rip)
    fldt 3360(%rdx)
    fldt 3376(%rdx)
    fyl2xp1
    fnstsw 11168(%rdi)
    fstpt 11184(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3392(%rdx)
    fldt 3408(%rdx)
    fyl2x
    fnstsw 11216(%rdi)
    fstpt 11232(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3424(%rdx)
    fldt 3440(%rdx)
    fyl2x
    fnstsw 11264(%rdi)
    fstpt 11280(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3456(%rdx)
    fldt 3472(%rdx)
    fyl2xp1
    fnstsw 11312(%rdi)
    fstpt 11328(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3488(%rdx)
    fldt 3504(%rdx)
    fyl2xp1
    fnstsw 11360(%rdi)
    fstpt 11376(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3520(%rdx)
    fldt 3536(%rdx)
    fyl2xp1
    fnstsw 11408(%rdi)
    fstpt 11424(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3552(%rdx)
    fldt 3568(%rdx)
    fyl2x
    fnstsw 11456(%rdi)
    fstpt 11472(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3584(%rdx)
    fldt 3600(%rdx)
    fyl2x
    fnstsw 11504(%rdi)
    fstpt 11520(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3616(%rdx)
    fldt 3632(%rdx)
    fyl2xp1
    fnstsw 11552(%rdi)
    fstpt 11568(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3648(%rdx)
    fldt 3664(%rdx)
    fyl2xp1
    fnstsw 11600(%rdi)
    fstpt 11616(%rdi)
    fninit
    fldcw cw077f(%rip)
    fldt 3680(%rdx)
    fldt 3696(%rdx)
    fyl2xp1
    fnstsw 11648(%rdi)
    fstpt 11664(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3712(%rdx)
    fldt 3728(%rdx)
    fyl2x
    fnstsw 11696(%rdi)
    fstpt 11712(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3744(%rdx)
    fldt 3760(%rdx)
    fyl2x
    fnstsw 11744(%rdi)
    fstpt 11760(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3776(%rdx)
    fldt 3792(%rdx)
    fyl2xp1
    fnstsw 11792(%rdi)
    fstpt 11808(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3808(%rdx)
    fldt 3824(%rdx)
    fyl2xp1
    fnstsw 11840(%rdi)
    fstpt 11856(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3840(%rdx)
    fldt 3856(%rdx)
    fyl2xp1
    fnstsw 11888(%rdi)
    fstpt 11904(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3872(%rdx)
    fldt 3888(%rdx)
    fyl2x
    fnstsw 11936(%rdi)
    fstpt 11952(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3904(%rdx)
    fldt 3920(%rdx)
    fyl2x
    fnstsw 11984(%rdi)
    fstpt 12000(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3936(%rdx)
    fldt 3952(%rdx)
    fyl2xp1
    fnstsw 12032(%rdi)
    fstpt 12048(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 3968(%rdx)
    fldt 3984(%rdx)
    fyl2xp1
    fnstsw 12080(%rdi)
    fstpt 12096(%rdi)
    fninit
    fldcw cw0b7f(%rip)
    fldt 4000(%rdx)
    fldt 4016(%rdx)
    fyl2xp1
    fnstsw 12128(%rdi)
    fstpt 12144(%rdi)
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
.balign 16
t:
    .quad 0xc000000000000000
    .short 0x3fff
    .zero 6
    .quad 0xc000000000000000
    .short 0xbfff
    .zero 6
    .quad 0x8000000000000000
    .short 0x4000
    .zero 6
    .quad 0x8000000000000000
    .short 0xc000
    .zero 6
    .quad 0xeccccccccccccccd
    .short 0x4000
    .zero 6
    .quad 0xeccccccccccccccd
    .short 0xc000
    .zero 6
    .quad 0xfa00000000000000
    .short 0x4008
    .zero 6
    .quad 0xfa00000000000000
    .short 0xc008
    .zero 6
    .quad 0xfffd333333333333
    .short 0x400c
    .zero 6
    .quad 0xfffd333333333333
    .short 0xc00c
    .zero 6
    .quad 0xfffe666666666666
    .short 0x400c
    .zero 6
    .quad 0xfffe666666666666
    .short 0xc00c
    .zero 6
    .quad 0x8000000000000000
    .short 0x403e
    .zero 6
    .quad 0x8000000000000000
    .short 0xc03e
    .zero 6
    .quad 0x8000000000000000
    .short 0x4063
    .zero 6
    .quad 0x8000000000000000
    .short 0xc063
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x5f3f
    .zero 6
    .quad 0x8000000000000000
    .short 0xdf3f
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x8000000000000001
    .short 0xbfff
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3fff
    .zero 6
    .quad 0xffffffffffffffff
    .short 0xbfff
    .zero 6
    .quad 0x8000000000000000
    .short 0x3fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xbfff
    .zero 6
    .quad 0xc000000000000000
    .short 0x3ffe
    .zero 6
    .quad 0xc000000000000000
    .short 0xbffe
    .zero 6
    .quad 0xaaaaaaaaaaaaaaab
    .short 0x3ffd
    .zero 6
    .quad 0xaaaaaaaaaaaaaaab
    .short 0xbffd
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0x3fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0xbfff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x999999999999999a
    .short 0x3ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x4000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0xc000000000000000
    .short 0xc000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x017f
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffb
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x017f
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffb
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x017f
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffb
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x017f
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffb
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0xfffe
    .zero 6
    .quad 0x8000000000000000
    .short 0x7ffd
    .zero 6
    .quad 0x8000000000000000
    .short 0x3fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xbfff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0x3fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0xbfff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x7fffffffffffffff
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x7fffffffffffffff
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x7fffffffffffffff
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x7fffffffffffffff
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x3fff
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x4000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x3ffe
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x400d
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0x403d
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x4000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x4000
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x400d
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x403f
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7e7f
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0xc000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000000
    .short 0x0000
    .zero 6
    .quad 0x8000000000000000
    .short 0x7fff
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x8000000000000000
    .short 0xffff
    .zero 6
    .quad 0x0000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x8000000000000001
    .short 0x3fff
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0xffffffffffffffff
    .short 0x3ffe
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x8000
    .zero 6
    .quad 0x4000000000000000
    .short 0x8000
    .zero 6
    .quad 0x0000000000000001
    .short 0x0000
    .zero 6
cw077f:
    .short 0x077f
cw0b7f:
    .short 0x0b7f
cw0f7f:
    .short 0x0f7f
