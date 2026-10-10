.text
.globl _start
_start:
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $0,%bl
    pushfq; pop %rax; mov %rax,0(%rdi); mov %rbx,8(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $0,%bl
    pushfq; pop %rax; mov %rax,16(%rdi); mov %rbx,24(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $1,%bl
    pushfq; pop %rax; mov %rax,32(%rdi); mov %rbx,40(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $1,%bl
    pushfq; pop %rax; mov %rax,48(%rdi); mov %rbx,56(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $2,%bl
    pushfq; pop %rax; mov %rax,64(%rdi); mov %rbx,72(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $2,%bl
    pushfq; pop %rax; mov %rax,80(%rdi); mov %rbx,88(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $8,%bl
    pushfq; pop %rax; mov %rax,96(%rdi); mov %rbx,104(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $8,%bl
    pushfq; pop %rax; mov %rax,112(%rdi); mov %rbx,120(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $9,%bl
    pushfq; pop %rax; mov %rax,128(%rdi); mov %rbx,136(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $9,%bl
    pushfq; pop %rax; mov %rax,144(%rdi); mov %rbx,152(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $16,%bl
    pushfq; pop %rax; mov %rax,160(%rdi); mov %rbx,168(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $16,%bl
    pushfq; pop %rax; mov %rax,176(%rdi); mov %rbx,184(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $17,%bl
    pushfq; pop %rax; mov %rax,192(%rdi); mov %rbx,200(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $17,%bl
    pushfq; pop %rax; mov %rax,208(%rdi); mov %rbx,216(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $31,%bl
    pushfq; pop %rax; mov %rax,224(%rdi); mov %rbx,232(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $31,%bl
    pushfq; pop %rax; mov %rax,240(%rdi); mov %rbx,248(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $32,%bl
    pushfq; pop %rax; mov %rax,256(%rdi); mov %rbx,264(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $32,%bl
    pushfq; pop %rax; mov %rax,272(%rdi); mov %rbx,280(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $33,%bl
    pushfq; pop %rax; mov %rax,288(%rdi); mov %rbx,296(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $33,%bl
    pushfq; pop %rax; mov %rax,304(%rdi); mov %rbx,312(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $63,%bl
    pushfq; pop %rax; mov %rax,320(%rdi); mov %rbx,328(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $63,%bl
    pushfq; pop %rax; mov %rax,336(%rdi); mov %rbx,344(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $64,%bl
    pushfq; pop %rax; mov %rax,352(%rdi); mov %rbx,360(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $64,%bl
    pushfq; pop %rax; mov %rax,368(%rdi); mov %rbx,376(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $0,%bx
    pushfq; pop %rax; mov %rax,384(%rdi); mov %rbx,392(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $0,%bx
    pushfq; pop %rax; mov %rax,400(%rdi); mov %rbx,408(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $1,%bx
    pushfq; pop %rax; mov %rax,416(%rdi); mov %rbx,424(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $1,%bx
    pushfq; pop %rax; mov %rax,432(%rdi); mov %rbx,440(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $2,%bx
    pushfq; pop %rax; mov %rax,448(%rdi); mov %rbx,456(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $2,%bx
    pushfq; pop %rax; mov %rax,464(%rdi); mov %rbx,472(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $8,%bx
    pushfq; pop %rax; mov %rax,480(%rdi); mov %rbx,488(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $8,%bx
    pushfq; pop %rax; mov %rax,496(%rdi); mov %rbx,504(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $9,%bx
    pushfq; pop %rax; mov %rax,512(%rdi); mov %rbx,520(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $9,%bx
    pushfq; pop %rax; mov %rax,528(%rdi); mov %rbx,536(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $16,%bx
    pushfq; pop %rax; mov %rax,544(%rdi); mov %rbx,552(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $16,%bx
    pushfq; pop %rax; mov %rax,560(%rdi); mov %rbx,568(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $17,%bx
    pushfq; pop %rax; mov %rax,576(%rdi); mov %rbx,584(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $17,%bx
    pushfq; pop %rax; mov %rax,592(%rdi); mov %rbx,600(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $31,%bx
    pushfq; pop %rax; mov %rax,608(%rdi); mov %rbx,616(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $31,%bx
    pushfq; pop %rax; mov %rax,624(%rdi); mov %rbx,632(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $32,%bx
    pushfq; pop %rax; mov %rax,640(%rdi); mov %rbx,648(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $32,%bx
    pushfq; pop %rax; mov %rax,656(%rdi); mov %rbx,664(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $33,%bx
    pushfq; pop %rax; mov %rax,672(%rdi); mov %rbx,680(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $33,%bx
    pushfq; pop %rax; mov %rax,688(%rdi); mov %rbx,696(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $63,%bx
    pushfq; pop %rax; mov %rax,704(%rdi); mov %rbx,712(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $63,%bx
    pushfq; pop %rax; mov %rax,720(%rdi); mov %rbx,728(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $64,%bx
    pushfq; pop %rax; mov %rax,736(%rdi); mov %rbx,744(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $64,%bx
    pushfq; pop %rax; mov %rax,752(%rdi); mov %rbx,760(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $0,%ebx
    pushfq; pop %rax; mov %rax,768(%rdi); mov %rbx,776(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $0,%ebx
    pushfq; pop %rax; mov %rax,784(%rdi); mov %rbx,792(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $1,%ebx
    pushfq; pop %rax; mov %rax,800(%rdi); mov %rbx,808(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $1,%ebx
    pushfq; pop %rax; mov %rax,816(%rdi); mov %rbx,824(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $2,%ebx
    pushfq; pop %rax; mov %rax,832(%rdi); mov %rbx,840(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $2,%ebx
    pushfq; pop %rax; mov %rax,848(%rdi); mov %rbx,856(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $8,%ebx
    pushfq; pop %rax; mov %rax,864(%rdi); mov %rbx,872(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $8,%ebx
    pushfq; pop %rax; mov %rax,880(%rdi); mov %rbx,888(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $9,%ebx
    pushfq; pop %rax; mov %rax,896(%rdi); mov %rbx,904(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $9,%ebx
    pushfq; pop %rax; mov %rax,912(%rdi); mov %rbx,920(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $16,%ebx
    pushfq; pop %rax; mov %rax,928(%rdi); mov %rbx,936(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $16,%ebx
    pushfq; pop %rax; mov %rax,944(%rdi); mov %rbx,952(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $17,%ebx
    pushfq; pop %rax; mov %rax,960(%rdi); mov %rbx,968(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $17,%ebx
    pushfq; pop %rax; mov %rax,976(%rdi); mov %rbx,984(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $31,%ebx
    pushfq; pop %rax; mov %rax,992(%rdi); mov %rbx,1000(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $31,%ebx
    pushfq; pop %rax; mov %rax,1008(%rdi); mov %rbx,1016(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $32,%ebx
    pushfq; pop %rax; mov %rax,1024(%rdi); mov %rbx,1032(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $32,%ebx
    pushfq; pop %rax; mov %rax,1040(%rdi); mov %rbx,1048(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $33,%ebx
    pushfq; pop %rax; mov %rax,1056(%rdi); mov %rbx,1064(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $33,%ebx
    pushfq; pop %rax; mov %rax,1072(%rdi); mov %rbx,1080(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $63,%ebx
    pushfq; pop %rax; mov %rax,1088(%rdi); mov %rbx,1096(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $63,%ebx
    pushfq; pop %rax; mov %rax,1104(%rdi); mov %rbx,1112(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $64,%ebx
    pushfq; pop %rax; mov %rax,1120(%rdi); mov %rbx,1128(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $64,%ebx
    pushfq; pop %rax; mov %rax,1136(%rdi); mov %rbx,1144(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $0,%rbx
    pushfq; pop %rax; mov %rax,1152(%rdi); mov %rbx,1160(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $0,%rbx
    pushfq; pop %rax; mov %rax,1168(%rdi); mov %rbx,1176(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $1,%rbx
    pushfq; pop %rax; mov %rax,1184(%rdi); mov %rbx,1192(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $1,%rbx
    pushfq; pop %rax; mov %rax,1200(%rdi); mov %rbx,1208(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $2,%rbx
    pushfq; pop %rax; mov %rax,1216(%rdi); mov %rbx,1224(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $2,%rbx
    pushfq; pop %rax; mov %rax,1232(%rdi); mov %rbx,1240(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $8,%rbx
    pushfq; pop %rax; mov %rax,1248(%rdi); mov %rbx,1256(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $8,%rbx
    pushfq; pop %rax; mov %rax,1264(%rdi); mov %rbx,1272(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $9,%rbx
    pushfq; pop %rax; mov %rax,1280(%rdi); mov %rbx,1288(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $9,%rbx
    pushfq; pop %rax; mov %rax,1296(%rdi); mov %rbx,1304(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $16,%rbx
    pushfq; pop %rax; mov %rax,1312(%rdi); mov %rbx,1320(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $16,%rbx
    pushfq; pop %rax; mov %rax,1328(%rdi); mov %rbx,1336(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $17,%rbx
    pushfq; pop %rax; mov %rax,1344(%rdi); mov %rbx,1352(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $17,%rbx
    pushfq; pop %rax; mov %rax,1360(%rdi); mov %rbx,1368(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $31,%rbx
    pushfq; pop %rax; mov %rax,1376(%rdi); mov %rbx,1384(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $31,%rbx
    pushfq; pop %rax; mov %rax,1392(%rdi); mov %rbx,1400(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $32,%rbx
    pushfq; pop %rax; mov %rax,1408(%rdi); mov %rbx,1416(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $32,%rbx
    pushfq; pop %rax; mov %rax,1424(%rdi); mov %rbx,1432(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $33,%rbx
    pushfq; pop %rax; mov %rax,1440(%rdi); mov %rbx,1448(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $33,%rbx
    pushfq; pop %rax; mov %rax,1456(%rdi); mov %rbx,1464(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $63,%rbx
    pushfq; pop %rax; mov %rax,1472(%rdi); mov %rbx,1480(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $63,%rbx
    pushfq; pop %rax; mov %rax,1488(%rdi); mov %rbx,1496(%rdi)
    mov $0x8123456789abcdef,%rbx
    clc
    rcr $64,%rbx
    pushfq; pop %rax; mov %rax,1504(%rdi); mov %rbx,1512(%rdi)
    mov $0x8123456789abcdef,%rbx
    stc
    rcr $64,%rbx
    pushfq; pop %rax; mov %rax,1520(%rdi); mov %rbx,1528(%rdi)
    ret
