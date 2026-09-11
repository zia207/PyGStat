C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C                                                                      %
C Copyright (C) 2003, Statios Software and Services Incorporated.  All %
C rights reserved.                                                     %
C                                                                      %
C This program has been modified from the one distributed in 1996 (see %
C below).  This version is also distributed in the hope that it will   %
C be useful, but WITHOUT ANY WARRANTY. Compiled programs based on this %
C code may be redistributed without restriction; however, this code is %
C for one developer only. Each developer or user of this source code   %
C must purchase a separate copy from Statios.                          %
C                                                                      %
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C                                                                      %
C Copyright (C) 1996, The Board of Trustees of the Leland Stanford     %
C Junior University.  All rights reserved.                             %
C                                                                      %
C The programs in GSLIB are distributed in the hope that they will be  %
C useful, but WITHOUT ANY WARRANTY.  No author or distributor accepts  %
C responsibility to anyone for the consequences of using them or for   %
C whether they serve any particular purpose or work at all, unless he  %
C says so in writing.  Everyone is granted permission to copy, modify  %
C and redistribute the programs in GSLIB, but only under the condition %
C that this notice and the above copyright notice remain intact.       %
C                                                                      %
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      program main
c-----------------------------------------------------------------------
c
c               Calibration for Markov/Bayes Simulation
c               ***************************************
c
c This program calibrates a set of primary and secondary data for input
c to the ``mbsim'' program.  Collocated primary (u) and secondary (v)
c samples are input data and the output is the local prior cdfs for the
c primary variable (u) given that the secondary variable (v) belongs to
c specific classes.  Other calibration information is also written
c to the output file.
c
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example bicalib.par)
c
c
c
c
c Original: Hua Zhu                                  Date:     July 1990
c Modified: Clayton V. Deutsch                                 1990-1999
c-----------------------------------------------------------------------
      use       msflib
c
c Fixed Parameters:
c
      parameter(MAXLEN=512,VERSION=3.000)
c
c Variable declaration:
c
      real      var(20)
      integer   nd,ncutu,ncutv,test
      character datafl*512,calfl*512,outfl*512,repfl*512,mbsimfl*512,
     +          str*512,strlin*512
      logical   testfl
c
c Declaration of dynamic arrays:
c
      real, allocatable :: u(:),v(:),wt(:),ucut(:),vcut(:),
     +      fract(:),lcdf(:),pdf(:,:),yx(:,:),soft(:,:,:),em(:,:),
     +      vm(:,:),softw(:,:,:),b(:)
      integer, allocatable :: nm(:,:)
c
c Input/Output units used:
c
      data      lin/1/,lout/2/,lrep/3/,lmb/4/
c
c Note VERSION number before anything else:
c
      write(*,9999) VERSION
 9999 format(/' BICALIB Version: ',f5.3/)
c
c Get the name of the parameter file - try the default name if no input:
c
      do i=1,512
            str(i:i) = ' '
      end do
      call getarg(1,str)
      if(str(1:1).eq.' ')then
            write(*,*) 'Which parameter file do you want to use?'
            read (*,'(a)') str
      end if
      if(str(1:1).eq.' ') str(1:20) = 'bicalib.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'bicalib.par         ') then
                  write(*,*) '        creating a blank parameter file'
                  call makepar
                  write(*,*)
            end if
            stop
      endif
      open(lin,file=str,status='OLD')
c
c Find Start of Parameters:
c
 1    read(lin,'(a4)',end=98) str(1:4)
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c

      read(lin,'(a512)',err=98) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=98) isec
      write(*,*) ' column for secondary variable = ',isec

      read(lin,'(a512)',err=98) calfl
      call chknam(calfl,512)
      write(*,*) ' data file = ',calfl(1:40)

      read(lin,*,err=98) ivu,ivv,iwt
      write(*,*) ' column for prim, sec, wt = ',ivu,ivv,iwt

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,'(a512)',err=98) mbsimfl
      call chknam(mbsimfl,512)
      write(*,*) ' output file for SISIM = ',mbsimfl(1:40)

      read(lin,'(a512)',err=98) repfl
      call chknam(repfl,512)
      write(*,*) ' report file = ',repfl(1:40)

      read(lin,*,err=98) ncutu
      write(*,*) ' # of primary cutoffs = ',ncutu
      
c
c Set paramaeters from parameter file:
c
      MXUCUT = ncutu
c
c Allocate the needed memory.
c
      allocate(ucut(0:MXUCUT+1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
c
      read(lin,*,err=98) (ucut(i),i=1,ncutu)
      write(*,*) ' primary cutoffs = ',(ucut(i),i=1,ncutu)

      read(lin,*,err=98) ncutv
      write(*,*) ' # of sec cutoffs = ',ncutv

      MXVCUT = ncutv
c
c Allocate the needed memory.
c
      allocate(vcut(0:MXVCUT+1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
c
      read(lin,*,err=98) (vcut(i),i=1,ncutv)
      write(*,*) ' secondary cutoffs = ',(vcut(i),i=1,ncutv)

      close(lin)
c
c Check to make sure the calibration data file exists:
c
      inquire(file=calfl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR file ',datafl,' does not exist!'
            stop
      endif
c
c The calibration file exists so open the file, read in the header
c information, and find MAXDAT parameter.
c
      write(*,*)
      write(*,*) 'Reading the calibration data'
      open(lin,file=calfl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*,err=99)
      end do
      maxdat = 0
 21   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      if(var(ivu).lt.tmin.or.var(ivu).ge.tmax) go to 21
      if(var(ivv).lt.tmin.or.var(ivv).ge.tmax) go to 21
      maxdat = maxdat + 1
      go to 21
 44   continue
c
c Initialize for some statistics:
c
      avgu = 0.0
      avgv = 0.0
      ssqu = 0.0
      ssqv = 0.0
      umin = 1.0e10
      vmin = 1.0e10
      umax =-1.0e10
      vmax =-1.0e10
      sum  = 0.0
c
c Allocate the needed memory.
c
      allocate(u(maxdat),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(v(maxdat),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(wt(maxdat),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(fract(MXUCUT),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(lcdf(MXUCUT),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(pdf(MXVCUT+1,MXUCUT+1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(yx(MXVCUT+1,MXUCUT+1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(soft(MXUCUT,0:1,maxdat),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(em(MXUCUT,0:1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(vm(MXUCUT,0:1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(softw(MXUCUT,0:1,maxdat),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(b(MXUCUT),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
      allocate(nm(MXUCUT,0:1),stat=test)
      if(test.ne.0)then
            write(*,*)'Allocation failed.'
            stop
      endif
c
c Read in as much data as the allocated storage will allow:
c
      rewind(lin)
      read(lin,*,err=99)
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*,err=99)
      end do
      nd = 0
 3    read(lin,*,end=4,err=99)(var(j),j=1,nvari)
      if(var(ivu).lt.tmin.or.var(ivu).ge.tmax) go to 3
      if(var(ivv).lt.tmin.or.var(ivv).ge.tmax) go to 3
      nd = nd + 1
      u(nd)  = var(ivu)
      v(nd)  = var(ivv)
      wt(nd) = 1.0
      if(iwt.gt.0) wt(nd) = var(iwt)
      sum  = sum  + wt(nd)
      avgu = avgu + u(nd)*wt(nd)
      avgv = avgv + v(nd)*wt(nd)
      ssqv = ssqv + v(nd)*v(nd)*wt(nd)
      ssqu = ssqu + u(nd)*u(nd)*wt(nd)
      if(u(nd).lt.umin) umin = u(nd)
      if(v(nd).lt.vmin) vmin = v(nd)
      if(u(nd).gt.umax) umax = u(nd)
      if(v(nd).gt.vmax) vmax = v(nd)
      go to 3
 4    close(lin)
c
c There has to be at least two data to go ahead with calibration:
c
      if(nd.le.1.or.sum.le.0.001) then
            write(*,*) 'TOO FEW DATA to go ahead = ',nd
            write(*,*) '          sum of weights = ',sum
            stop
      endif
      write(*,*)
      write(*,*) 'Calculating the calibration parameters'
      xd = 1.0 / real(sum)
      open(lrep,file=repfl,status='UNKNOWN')
      open(lmb,file=mbsimfl,status='UNKNOWN')
c
c Compute the averages and variances as an error check for the user:
c
      avgu = avgu * xd
      avgv = avgv * xd
      ssqu = ssqu * xd - avgu * avgu
      ssqv = ssqv * xd - avgv * avgv
      write(lrep,*)
      write(lrep,*) '                 MARKOV-BAYES CALIBRATION REPORT'
      write(lrep,*) '                 *******************************'
      write(lrep,*)
      write(lrep,*) '      Number of pairs retained     = ',nd
      write(lrep,*)
      write(lrep,*) '      Primary variable:   average  = ',avgu
      write(lrep,*) '                          variance = ',ssqu
      write(lrep,*) '                          minimum  = ',umin
      write(lrep,*) '                          maximum  = ',umax
      write(lrep,*)
      write(lrep,*) '      Secondary variable: average  = ',avgv
      write(lrep,*) '                          variance = ',ssqv
      write(lrep,*) '                          minimum  = ',vmin
      write(lrep,*) '                          maximum  = ',vmax
      write(lrep,*)
c
c Establish lower and upper bounds:
c
      ucut(0)       = umin - 1. - abs(umax)
      ucut(ncutu+1) = umax + abs(umax+1.)
      vcut(0)       = vmin - 1. - abs(vmax)
      vcut(ncutv+1) = vmax + abs(vmax+1.)
      do i=1,ncutu
            fract(i)  = 0.0
      end do
c
c Calculate the indicator mean for each cutoff:
c
      do i=1,nd
            do j=1,ncutu
                  if((u(i).gt.ucut(j-1)).and.(u(i).le.ucut(j)))then
                         fract(j)=fract(j)+wt(i)
                         go to 2
                  endif
            end do
 2          continue
      end do
c
c Turn fract() into a cdf:
c
      i  = 1
      fract(i) = fract(i) * xd
      write(lrep,*)  'Cutoffs on Primary Variable'
      write(lrep,100) i,ucut(i),fract(i)
      do i=2,ncutu
            fract(i) = fract(i-1) + fract(i) * xd
            write(lrep,100) i,ucut(i),fract(i)
      end do
 100  format('  U cutoff ',i2,' cutoff = ',f12.4,' cdf = ',f8.5)
      write(lrep,*)
c
c Compute the pdf table:
c
      do k=1,ncutv+1
            do j=1,ncutu+1
                  pdf(k,j) = 0.0
            end do
      end do
      do i=1,nd
             do k=1,ncutv+1
                if((v(i).gt.vcut(k-1)).and.(v(i).le.vcut(k)))then
                   do j=1,ncutu+1
                      if((u(i).gt.ucut(j-1)).and.(u(i).le.ucut(j)))then
                            pdf(k,j) = pdf(k,j) + wt(i)
                            go to 6
                      endif
                   end do
                endif
             end do
 6           continue
       end do
c
c Turn the bivariate pdf into conditional cdf, i.e. the local prior cdf
c
      do i=1,ncutv+1
            do j=1,ncutu+1
                  pdf(i,j) = pdf(i,j) * xd
            end do
      end do
c
c Write out the bivariate pdf table: 
c
      write(lrep,*) 'Number within each bivariate (u,v) class:'
      write(lrep,101)(ucut(j),j=1,ncutu)
 101  format(10x,9(1x,f8.2),
     +     /1x,'______________________________________________________')
      do i=1,ncutv
            write(lrep,102) vcut(i),(pdf(i,j),j=1,ncutu+1)
 102        format(1x,f9.2,' | ', 10(1x,f8.4) )
      end do
      write(lrep,103)(pdf(ncutv+1,j),j=1,ncutu+1)
 103  format('      Max.',' | ', 10(1x,f8.0) )
      write(lrep,*)
c
c Loop over all the v cutoffs:
c
      do i=1,ncutv+1
            cum = 0.0
            do j=1,ncutu+1
                   cum = cum + pdf(i,j)
            end do
            if(cum.gt.0.0) then
                  do j=1,ncutu+1
                        pdf(i,j) = pdf(i,j) / cum
                  end do
            endif
            do j=1,ncutu+1
                  yx(i,j) = 0.0
                  do k=1,j
                        yx(i,j) = yx(i,j) + pdf(i,k)
                  end do
            end do
      end do
c
c Write out the local prior cdf table:
c
      write(lrep,*)'The cumulative frequency (local prior cdf) table:' 
      write(lmb,* )'Thresholds for secondary variable'
      write(lmb,* ) ncutv
      do i=1,ncutv
            write(lmb,*) vcut(i)
      end do
      write(lmb,* )'The local prior distribution table:'
      do i=1,ncutv+1
            write(lrep,110) (yx(i,j),j=1,ncutu+1)
            write(lmb,110)  (yx(i,j),j=1,ncutu)
      end do
 110  format(20(1x,f7.5))
      write(lrep,*)
      write(lmb,*) 'The calibration parameters B(i): '
c
c Calculate the calibration parameters from the local prior cdf table
c and the input data:  First, initialize counters:
c
      do j=1,ncutu
            do k=0,1
                  nm(j,k)   = 0
                  em(j,k)   = 0.0
                  vm(j,k)   = 0.0
                  do n=1,nd
                        soft(j,k,n)  = 0.0
                        softw(j,k,n) = 0.0
                  end do
            end do
      end do
c
c MAIN Loop over all the u cutoffs to sort out the classification of
c all the data:
c
      do j=1,ncutu
c
c Now, loop over all the data:
c
            do i=1,nd
                  if(u(i).le.ucut(j))then
                        do k=1,ncutv+1
                              if((v(i).gt.vcut(k-1)).and.
     +                           (v(i).le.vcut(k)))  then
                                    nm(j,1)            = nm(j,1) + 1
                                    soft(j,1,nm(j,1))  = yx(k,j)
                                    softw(j,1,nm(j,1)) = wt(i)
                                    go to 11
                              endif
                        end do
                 else
                       do k=1,ncutv+1
                             if((v(i).gt.vcut(k-1)).and.
     +                          (v(i).le.vcut(k)))  then
                                   nm(j,0)            = nm(j,0) + 1
                                   soft(j,0,nm(j,0))  = yx(k,j)
                                   softw(j,0,nm(j,0)) = wt(i)
                                   go to 11
                             endif
                       end do
                 endif
 11              continue
           end do
      end do
c
c Now, get some statistics on the calibration.  Loop over all the u
c cutoffs and then the two classication categories:
c
      do j=1,ncutu
            do k=0,1
                  if(nm(j,k).ge.1) then
                        em(j,k) = 0.0
                        sum     = 0.0
                        do i=1,nm(j,k)
                              em(j,k) = em(j,k) + soft(j,k,i)*
     +                                            softw(j,k,i)
                              sum     = sum     + softw(j,k,i)
                        end do
                        em(j,k) = em(j,k) / max(real(sum),0.000001)
                        vm(j,k) = 0.0
                        sum     = 0.0
                        do i=1,nm(j,k)
                              dev     = soft(j,k,i) - em(j,k)
                              vm(j,k) = vm(j,k) + dev*dev*
     +                                            softw(j,k,i)
                              sum     = sum     + softw(j,k,i)
                        end do
                        vm(j,k) = vm(j,k) / max(real(sum),0.000001)
                  endif
            end do
      end do
c
c Summarize the results:  First the statistics, then B(z),
c
      write(lrep,120)
 120  format('cutoff,total#,mean,Variance') 
      do k=1,0,-1
            if(k.eq.1) write(lrep,*)' for U(x) <= cutoff '
            if(k.eq.0) write(lrep,*)' for U(x) >  cutoff '
            do j=1,ncutu
                  write(lrep,121) ucut(j),nm(j,k),em(j,k),vm(j,k)
 121              format(1x,f8.2,1x,i5,' | ', 2(1x,f9.5) )
            end do
      end do
      write(lrep,*)
c
c Hardness coefficients:
c
      write(lrep,*)'B(i) values:'
      do j=1,ncutu
            b(j)   = em(j,1) - em(j,0)
            if(nm(j,1).lt.1.or.nm(j,0).lt.1) b(j) = -9.0
            write(lrep,'(f7.4)') b(j)
            write(lmb,'(f7.4)') b(j)
      end do
      close(lrep)
c
c NOW, read through input data file and append the soft indicator data:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) go to 77
      open(lin,file=datafl,status='OLD')
      open(lout,file=outfl,status='UNKNOWN')
      write(*,*)
      write(*,*) 'Reading through the data file'
c
c Read the header off the data file and prepare the output files:
c
      read(lin,'(a40)',err=97) str
      write(lout,'(a40)') str
      read(lin,*,err=97) nvari
      write(lout,'(i3)') nvari+ncutu
      do i=1,nvari
            read(lin,'(a40)',err=97) str
            write(lout,'(a40)')      str
      end do
      do i=1,ncutu
            write(lout,131) ucut(i)
 131        format('Primary Threshold ',f12.4)
      end do
c
c Loop through all observations in the input file:
c
 40   read(lin,*,end=41,err=97) (var(j),j=1,nvari)
      vval = var(isec)
      if(vval.lt.tmin.or.vval.ge.tmax) then
            do i=1,ncutu
                  lcdf(i) = -9.0
            end do
      else
            do i=1,ncutv+1
                  if((vval.gt.vcut(i-1)).and.(vval.le.vcut(i))) then
                         icls = i
                         go to 42
                  endif
            end do
 42         continue
            do i=1,ncutu
                  lcdf(i) = yx(icls,i)
            end do
      end if
c
c Write (append) the local cdf to the output file:
c
      backspace lin
      read(lin,'(a)') strlin
      call strlen(strlin,MAXLEN,lostr)
      write(lout,'(a,1x,24(f8.4))') strlin(1:lostr),(lcdf(i),i=1,ncutu)
      go to 40
c
c Finished with all input records:
c
 41   continue
      close(lin)
      close(lout)
 77   continue
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' BICALIB Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in data file'
 98   stop 'ERROR in parameter file'
 99   stop 'ERROR in calibration data file'
      end



      subroutine makepar
c-----------------------------------------------------------------------
c
c                      Write a Parameter File
c                      **********************
c
c
c
c-----------------------------------------------------------------------
      lun = 99
      open(lun,file='bicalib.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for BICALIB',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/ydata.dat                 ',
     +       '-file with secondary data')
      write(lun,12)
 12   format('4                                 ',
     +       '-   column for secondary variable')
      write(lun,13)
 13   format('../data/cluster.dat               ',
     +       '-file with calibration scatterplot')
      write(lun,14)
 14   format('3  4  5                           ',
     +       '-   columns of pri, sec, and weight')
      write(lun,15)
 15   format('-1.0e21   1.0e21                  ',
     +       '-   trimming limits')
      write(lun,16)
 16   format('bicalib.out                       ',
     +       '-file for output data / distributions')
      write(lun,17)
 17   format('bicalib.cal                       ',
     +       '-file for output calibration (SISIM)')
      write(lun,18)
 18   format('bicalib.rep                       ',
     +       '-file for calibration report')
      write(lun,19)
 19   format('5                                 ',
     +       '-number of thresholds on primary')
      write(lun,20)
 20   format('0.50 1.00 2.50 5.00 10.0          ',
     +       '-   thresholds on primary')
      write(lun,21)
 21   format('5                                 ',
     +       '-number of thresholds on secondary')
      write(lun,22)
 22   format('0.50 1.00 2.50 5.00 10.0          ',
     +       '-   thresholds on secondary')

      close(lun)
      return
      end
