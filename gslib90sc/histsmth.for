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
c             Smooth a Histogram with Simulated Annealing
c             *******************************************
c
c Creates a smooth distribution honoring (optionally) the mean,
c variance, and quantiles from user supplied data.
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example histsmth.par)
c
c
c The output file will contain the smoothed distribution specified by
c two columns (a "z" value and corresponding "p" value)
c
c
c
c INPUT/OUTPUT Parameters:
c
c   datafl      file with data
c   ivr,iwt     columns for variable and weight
c   tmin,tmax   trimming limits
c   title       title
c   postfl      file for PostScript output
c   outfl       file for smoothed output
c   n,zmin,zmax smoothing limits: number, min, max
c   ilog        0=arithmetic, 1=log scaling
c   options     maxpert, reporting interval,min Obj,Seed
c   flags       1=y,0=n: mean,var,smth,qua
c   weights     scaling: mean,var,smth,qua
c   nhist       number of hist classes (for display only)
c   nsmooth     size of smoothing window
c   mean,var    mean and variance (-999=>data)
c   nqd         number of quantiles: defined from data
c   nqu         number of quantiles: defined by user
c   cdf()z()    user defined quantiles
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999 
c-----------------------------------------------------------------------
      include 'histsmth.inc'
c
c Read the parameters and reference data distribution:
c
      call readparm
c
c Smooth the histogram:
c
      call smooth
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' HISTSMTH Version: ',f5.3, ' Finished'/)
      stop
      end



      subroutine readparm
c-----------------------------------------------------------------------
c
c
c
c
c
c
c-----------------------------------------------------------------------
      use       msflib
      include 'histsmth.inc'

      parameter (KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
      common/iaco/ ixv(MAXOP1)
      real*8 acorni

      real, allocatable :: rcdf(:),rvr(:)

      real var(100)
      integer test

      character datafl*512,outfl*512,psfl*512,title*40,str*512,xlab*12
      logical   testfl
c
c Note VERSION number before anything else:
c
      lin  = 1
      lout = 2
      lps  = 4
      write(*,9999) VERSION
 9999 format(/' HISTSMTH Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'histsmth.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'histsmth.par        ') then
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
 1    read(lin,'(a4)',end=97) str(1:4)
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c

      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' reference data file = ',datafl(1:40)

      read(lin,*,err=97) ivr,iwt
      write(*,*) ' columns = ',ivr,iwt

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a40)',err=97) title 
      call chktitle(title,40)
      write(*,*) ' title = ',title

      read(lin,'(a512)',err=97) psfl 
      call chknam(psfl,512)
      write(*,*) ' output PostScript file = ',psfl(1:40)

      read(lin,*,err=97) nhist
      write(*,*) ' number of histogram classes = ',nhist
      if(nhist.gt.MAXCLS) stop 'nhist is too big'

      read(lin,'(a512)',err=97) outfl 
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) nclass,zmin,zmax
      write(*,*) ' data limits = ',nclass,zmin,zmax

      read(lin,*,err=97) ilog
      write(*,*) ' logarithmic scaling flag = ',ilog
      if(ilog.eq.1) then
            if(zmin.le.EPSLON) stop ' must have minimum value > 0'
            if(zmax.le.EPSLON) stop ' must have maximum value > 0'
            zmin = alog10(zmin)
            zmax = alog10(zmax)
      endif

      read(lin,*,err=97) xm,xr,omin,ixv(1)
      write(*,*) ' maxpert,report,omin = ',xm,xr,omin,ixv(1)
      do i=1,10000
            tzz = real(acorni(idum))
      end do

      ncut    = nclass - 1
      maxpert = int(xm*nclass)
      report  = int(xr*nclass)

      read(lin,*,err=97) i1,i2,i3,i4
      write(*,*) ' Components in objective func = ',i1,i2,i3,i4
                  lmean = .false.
                  lvari = .false.
                  lsmth = .false.
                  lquan = .false.
      if(i1.ge.1) lmean = .true.
      if(i2.ge.1) lvari = .true.
      if(i3.ge.1) lsmth = .true.
      if(i4.ge.1) lquan = .true.

      read(lin,*,err=97) sclmean,sclvari,sclsmth,sclquan
      write(*,*) ' scaling of objective func = ',sclmean,sclvari,
     +                                           sclsmth,sclquan
      totwt = sclmean + sclvari + sclsmth + sclquan
      if(totwt.le.0.0) stop 'Weights must be greater than zero'
      sclmean = sclmean / totwt
      sclvari = sclvari / totwt
      sclsmth = sclsmth / totwt
      sclquan = sclquan / totwt

      read(lin,*,err=97) nsmooth
      write(*,*) ' Size of smoothing half window = ',nsmooth
      if(nsmooth.gt.MAXSMT) stop 'nsmooth is too big'
      if(nsmooth.gt.int(0.25*nclass)) then
            write(*,*) 'ERROR: when nsmooth > 0.25 * nclass it is '
            write(*,*) '       difficult to smooth the distribution'
            stop
      end if
      do ismth=1,nclass
            smthfac(ismth) = 1.0 / real(2*nsmooth)
      end do
      do ismth=1,nsmooth
            smthfac(ismth) = 1.0 / real(nsmooth+ismth-1)
            smthfac(nclass-ismth+1) = smthfac(ismth)
      end do

      read(lin,*,err=97) mean,variance
      write(*,*) ' mean and variance = ',mean,variance
      if(ilog.eq.1.and.mean.ge.-990.0) then
            write(*,*) 'Must get mean from the data with log option'
            mean = -1000.0
      end if
      if(ilog.eq.1.and.variance.ge.-990.0) then
            write(*,*) 'Must get variance from the data with log option'
            variance = -1000.0
      end if

      read(lin,*,err=97) ndq
      write(*,*) ' number of data quantiles = ',ndq

      read(lin,*,err=97) nuq
      write(*,*) ' number of user quantiles = ',nuq
      do i=1,nuq
            read(lin,*,err=97) qval(i),tzz
            if(ilog.eq.1) tzz = alog10(max(tzz,EPSLON))
            uqind(i) = 1 + int((tzz-zmin)/(zmax-zmin)*nclass)
      end do

      close(lin)
c
c Read in the reference data distribution:
c
      nd    = 0
      nt    = 0
      tcdf  = 0
      gmean = 0.0
      gvar  = 0.0
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'No data file!'
            go to 4
      endif
c
c The first data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='UNKNOWN')
      read(lin,*,err=98)
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0
 222  read(lin,*,end=444,err=98)(var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 222
      maxdat = maxdat + 1
      go to 222
 444  continue
c
c Allocate the needed memory.
c
      allocate (rcdf(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error 1: Allocation failed due to ',
     +                       'insufficient memory!', test
            stop
      end if
      allocate (rvr(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error 1: Allocation failed due to ',
     +                       'insufficient memory!', test
            stop
      end if
c     
c  Now read the data for real:
c
      rewind(lin)
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'(a12)',err=98) str(1:12)
            if(i.eq.ivr) xlab = str(1:12)
      end do
c
c Read as much data as possible:
c
 3    read(lin,*,end=4,err=98) (val(j),j=1,nvari)
      if(ilog.eq.1) val(ivr) = alog10(max(val(ivr),EPSLON))
c
c Trim this data?
c
      if((zmin.lt.zmax).and.(val(ivr).lt.zmin.or.val(ivr).ge.zmax)) then
            nt = nt + 1
            go to 3
      endif
      if(iwt.ge.1) then
            if(val(iwt).le.EPSLON) then
                  nt = nt + 1
                  go to 3
            endif
      endif
c
c Accept this data:
c
      nd = nd + 1
      if(nd.gt.MAXDAT) then
            write(*,*) 'ERROR: exceeded available storage for data'
            write(*,*) '       have ',MAXDAT,' available'
            stop
      endif
      rvr(nd) = val(ivr)
      if(iwt.ge.1) then
            rcdf(nd) = val(iwt)
      else
            rcdf(nd) = 1.0
      endif
      tcdf  = tcdf  + rcdf(nd)
      gmean = gmean + rvr(nd)*rcdf(nd)
      gvar  = gvar  + rvr(nd)*rvr(nd)*rcdf(nd)
c
c Go back for another data:
c
      go to 3
 4    close(lin)
c
c If there are any data:
c
      if(nd.gt.0) then
            gmean = gmean / tcdf
            gvar  = gvar  / tcdf - (gmean*gmean)
c
c Sort the Reference Distribution and Check for error situation:
c
            call sortem(1,nd,rvr,1,rcdf,c,d,e,f,g,h)
            if(nd.le.1.or.tcdf.le.EPSLON) then
                  write(*,*) 'ERROR: too few data or too low weight'
                  stop
            endif
c
c Turn the (possibly weighted) distribution into a cdf that is useful:
c
            oldcp = 0.0
            cp    = 0.0
            tcdf  = 1.0 / tcdf
            do i=1,nd
                  cp      = cp + rcdf(i) * tcdf
                  rcdf(i) =(cp + oldcp) * 0.5
                  oldcp   = cp
            end do
c
c Get median and write some info to the screen:
c
            qq = 0.25
            call locate(rcdf,nd,1,nd,qq,j)
            glq = powint(rcdf(j),rcdf(j+1),rvr(j),rvr(j+1),qq,1.)
            qq = 0.5
            call locate(rcdf,nd,1,nd,qq,j)
            gmedian = powint(rcdf(j),rcdf(j+1),rvr(j),rvr(j+1),qq,1.)
            qq = 0.75
            call locate(rcdf,nd,1,nd,qq,j)
            guq = powint(rcdf(j),rcdf(j+1),rvr(j),rvr(j+1),qq,1.)

            write(*,101)    nd,nt,gmean,gvar,rvr(1),glq,gmedian,
     +                                              guq,rvr(nd)
 101        format(//,' HISTSMTH: number of data : ',8x,i4,/,
     +          '           number trimmed : ',8x,i4,/,
     +          '           mean           : ',f12.4,/,
     +          '           variance       : ',f12.4,//,
     +          '           minimum        : ',f12.4,/,
     +          '           lower quartile : ',f12.4,/,
     +          '           median         : ',f12.4,/,
     +          '           upper quartile : ',f12.4,/,
     +          '           maximum        : ',f12.4,/)
      end if
c
c Reset the mean and variance if the user has not specified them:
c
      if(mean    .lt.-990.0) mean     = gmean
      if(variance.lt.-990.0) variance = gvar
      if(zmax.lt.zmin) then
            zmin  = rvr(1)
            zmax  = rvr(nd)
            ltpar = zmin
            utpar = zmax
      end if
      zinc    = (zmax-zmin) / real(nclass-1)
c
c The "z" and "z^2" values:
c
      do i=1,nclass
            zz(i)  = zmin+real(i-1)*zinc
            zzs(i) = zz(i)*zz(i)
      end do
c
c Establish the quantiles:
c
      if(nd.le.0) then
            ndq = 0
            lquan = .false.
      end if
      qinc = 1.0 / real(ndq+1)
      do i=1,ndq
            nuq = nuq + 1
            if(nuq.gt.MAXSTA) stop 'too many quantiles'
            qval(nuq) = real(i) * qinc
            call locate(rcdf,nd,1,nd,qval(nuq),j)
            tzz = powint(rcdf(j),rcdf(j+1),rvr(j),rvr(j+1),qval(nuq),1.)
            uqind(nuq) = 1 + int((tzz-zmin)/(zmax-zmin)*nclass)
      end do
c
c Get the output file ready:
c
      open(lout,file=outfl,status='UNKNOWN')
      if(ilog.eq.1) then
            write(lout,110)
 110        format('Smooth Results',/,'3',/,'Z-value',/,
     +             'Log10 Z-value',/,'P-value')
      else
            write(lout,111)
 111        format('Smooth Results',/,'2',/,'Z-value',/,'P-value')
      end if
c
c Open the PostScript file:
c
      open(lps,file=psfl,status='UNKNOWN')
      write(lps,998) zmin,zmax,title,xlab
 998  format('%!PS                                 %    Remove     ',
     +    /, '90 234 translate 1.5 1.5 scale       %  these lines  ',
     +    /, '                                     % for EPSF file ',
     +    /, '%!PS-Adobe-3.0 EPSF-3.0',
     +    /, '%%BoundingBox: 0 0 360 360',
     +    /, '%%Creator: GSLIB',
     +    /, '%%Title:   Output from HISTSMTH',
     +    /, '%%CreationDate: 12/10/93',
     +    /, '%%EndComments',/,/,/,'%',/,'%',/,'%',/,
     +    /, '/gr{grestore} bind def /gs{gsave} bind def',
     +    /, '/m {moveto} def /l {lineto} def /r {rlineto} def',
     +    /, '/s {stroke} def /n {newpath} def /c {closepath} def',
     +    /, '/rtext{ dup stringwidth pop -1 div 0 rmoveto show } def',
     +    /, '/ctext{ dup stringwidth pop -2 div 0 rmoveto show } def',
     +    /, '/ltext{show} def  /bullet{ 6 0 360 arc c fill } def',
     +    /, '/wbullet{ 6 0 360 arc gs 0.8 setgray fill gr} def',
     +    /, '/tr{translate} bind def /sc{setrgbcolor} bind def',
     +   //, '0.9 0.9 scale 0.24 0.24 scale 50 100 translate',/,
     +       '%',/,'% Bottom Histogram Plot:',/,'%',/,
     +       '2.5 setlinewidth n 0 900 m 0 0 l 1250 0 l s ',/,
     +       '/Helvetica findfont 30 scalefont setfont',/, 
     +       '0  -60 m  (',f8.2,')  ctext 1200 -60 m  (',f8.2,') rtext',
     +     /,'/Helvetica-BoldOblique findfont 33 scalefont setfont',/, 
     +       '10  900 m  (',a40,')  ltext',/,
     +       '/Helvetica findfont 30 scalefont setfont',/, 
     +       '450 -60 m  (',a40')  ctext')
c
c Determine the histogram class structure:
c
      psmax = -999.0
      if(nd.gt.0) then
            dcl = (zmax-zmin)/real(nhist)
            do i=1,nhist
                  val(i) = 0
            end do
            do i=1,nd
                  ihist = 1 + int((rvr(i)-zmin)/dcl)
                  val(ihist) = val(ihist) + 1
            end do
            do i=1,nhist
                  val(i) = val(i) / real(nd)
            end do
c
c Write the histogram to the PostScript file:
c
            psmax = 0.0
            do icls=1,nhist
                  if(val(icls).gt.psmax) psmax = val(icls)
            end do
            do icls=1,nhist
                  zlo = real(icls-1)/real(nhist) *1200.0
                  zup = real(icls  )/real(nhist) *1200.0
                  pup = val(icls)   / psmax      * 800.0
                  write(lps,999) zlo,zlo,pup,zup,pup,zup
            end do
      end if
 999  format('n ',f6.1,' 0 m ',f6.1,1x,f6.1,' l ',f6.1,1x,f6.1,' l ',/,
     +       '  ',f6.1,' 0 l c gs 0.90 setgray fill gr s')
c
c END Main loop:
c
      return
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in data file!'
      end



      subroutine smooth
c-----------------------------------------------------------------------
c
c        Main subroutine that performs the Annealing Smoothing
c        *****************************************************
c
c
c
c
c
c-----------------------------------------------------------------------
      include 'histsmth.inc'
      real*8   acorni
c
c Initialize the p values:
c
      do i=1-nsmooth,nclass+nsmooth
            val(i) = 0.0
      end do
c
c Establish initial P values:
c
      sump = 0.0
      do i=1,nclass
            val(i) = real(acorni(idum))
            sump = sump + val(i)
      end do
      do i=1,nclass
            val(i) = val(i) / sump
      end do
c
c Initialize:
c
      objmean = 0.0
      objvari = 0.0
      objsmth = 0.0
      objquan = 0.0

      obtmean = 0.0
      obtvari = 0.0
      obtsmth = 0.0
      obtquan = 0.0

      delmean = 0.0
      delvari = 0.0
      delsmth = 0.0
      delquan = 0.0

      snomean = 0.0
      snovari = 0.0
      snosmth = 0.0
      snoquan = 0.0
c
c Establish initial objective function values:
c
      call initobj
c
c Establish weights by looping over a large number of random
c perturbations and keep track of how much each objective function
c component changes:
c
      do i=1,NUMSCL
            call getpert(ipert,jpert,del)
            call object(ipert,jpert,del)
            if(lmean) delmean  = delmean  + abs(objmean-obtmean)
            if(lvari) delvari  = delvari  + abs(objvari-obtvari)
            if(lsmth) delsmth  = delsmth  + abs(objsmth-obtsmth)
            if(lquan) delquan  = delquan  + abs(objquan-obtquan)
      end do
c
c Establish each component scaling.  The scaling is based on my
c scaling, user scaling, and the average absolute deviation based on
c NUMSCL perturbations:
c
      if(lmean.and.delmean.gt.0.0) then
            sclmean = 1.0 * sclmean / delmean
      else
            sclmean = 0.0
      end if
      if(lvari.and.delvari.gt.0.0) then
            sclvari = 1.0 * sclvari / delvari
      else
            sclvari = 0.0
      end if
      if(lsmth.and.delsmth.gt.0.0) then
            sclsmth = 5.0 * sclsmth / delsmth
      else
            sclsmth = 0.0
      end if
      if(lquan.and.delquan.gt.0.0) then
            sclquan = 4.0 * sclquan / delquan
      else
            sclquan = 0.0
      end if
      resc    = 1.0 / (  sclmean*objmean 
     +                 + sclvari*objvari
     +                 + sclsmth*objsmth
     +                 + sclquan*objquan )
      if(lmean) sclmean = resc * sclmean
      if(lvari) sclvari = resc * sclvari
      if(lsmth) sclsmth = resc * sclsmth
      if(lquan) sclquan = resc * sclquan
      if(lmean) snomean = 1.0 / objmean
      if(lvari) snovari = 1.0 / objvari
      if(lsmth) snosmth = 1.0 / objsmth
      if(lquan) snoquan = 1.0 / objquan
c
c Loop until convergence or the stopping number:
c
      naccept = 0
      do iswap=1,maxpert
c
c Report:
c
            if(iswap.eq.1.or.(int(iswap/report)*report).eq.iswap) then
                  write(*,996)    iswap,real(objmean) * snomean,
     +                                  real(objvari) * snovari,
     +                                  real(objsmth) * snosmth,
     +                                  real(objquan) * snoquan
 996              format(' Status at',i8:,' m=',f9.5,' v=',f9.5,
     +                                    ' s=',f9.5,' q=',f9.5)
            end if
c
c Perturb:
c
            call getpert(ipert,jpert,del)
c
c Update:
c
            obj    = objmean * sclmean
     +             + objvari * sclvari
     +             + objsmth * sclsmth
     +             + objquan * sclquan

            call object(ipert,jpert,del)

            objtry = obtmean * sclmean
     +             + obtvari * sclvari
     +             + obtsmth * sclsmth
     +             + obtquan * sclquan
c
c Decide whether or not to keep:
c
            if(objtry.lt.obj) then
                  val(ipert) = val(ipert) + del
                  val(jpert) = val(jpert) - del
                  objmean    = obtmean
                  objvari    = obtvari
                  objsmth    = obtsmth 
                  objquan    = obtquan

                  rmean= tmean
                  rvar = tvar 

                  if(lsmth) then
                        do i=-nsmooth,nsmooth
                              if(i.ne.0) then 
                                    loc = ipert + i
                                    if(loc.ge.1.and.loc.le.nclass)
     +                                  sval(loc) = svali(i)
                                    loc = jpert + i
                                    if(loc.ge.1.and.loc.le.nclass)
     +                                  sval(loc) = svalj(i)
                              end if
                        end do
                  end if
                  if(lquan) then
                        do iq=1,nuq
                              qact(iq) = qtry(iq)
                        end do
                  end if
            endif
c
c Go back for another perturbation?
c
            if(real(obj).lt.omin) go to 30
      end do
c
c Finished:
c
 30   continue
      write(*,996)    iswap,real(objmean) * snomean,
     +                      real(objvari) * snovari,
     +                      real(objsmth) * snosmth,
     +                      real(objquan) * snoquan
      call initobj
      write(*,996)    iswap,real(objmean) * snomean,
     +                      real(objvari) * snovari,
     +                      real(objsmth) * snosmth,
     +                      real(objquan) * snoquan
c
c Write the final values (to the output file and the PostScript file):
c
      write(lps,101)
 101  format('n 0 0 m')
      if(psmax.le.0.0) then
            do i=1,nclass
                  if(val(i).gt.psmax) psmax = val(i)
            end do
            nhist = nclass
      end if
      sclfac = 800.0 * real(nclass) / psmax / real(nhist)
      do i=1,nclass
            if(ilog.eq.1) then
                  write(lout,'(f10.4,1x,f10.4,1x,f9.6)') 
     +                  10**zz(i),zz(i),val(i)
            else
                  write(lout,'(f10.4,1x,f9.6)') zz(i),val(i)
            end if
            tzz = (real(zz(i))-zmin)/(zmax-zmin)*1200.0
            pp =  val(i) * sclfac
            write(lps,102) tzz,pp
 102        format(f6.1,1x,f6.1,' l')
      end do
      write(lps,103)
 103  format('s')
c
c Add a footer to the Postscript plot file:
c
      write(lps,997)
 997  format('%END OF POSTSCRIPT FILE',/,
     +       '1.111111 1.111111 scale 4.166667 4.166667 scale',/,/,
     +       '%%EOF',/,'showpage')
c
c Return:
c
      return
      end



      subroutine getpert(ipert,jpert,del)
c-----------------------------------------------------------------------
c
c                  Perturbation Mechanism
c                  **********************
c
c
c
c
c-----------------------------------------------------------------------
      include 'histsmth.inc'
      real*8   acorni
c
c Get two distant locations:
c
 2    ipert  = 1 + int(real(acorni(idum))*nclass)
      jpert  = 1 + int(real(acorni(idum))*nclass)
      if(abs(ipert-jpert).lt.(2*nsmooth+1)) go to 2
c
c decide how much to perturb:
c
      del  = real(acorni(idum)) * 0.05 * val(jpert)
c
c End Perturbation mechanism:
c
      return
      end



      subroutine initobj
c-----------------------------------------------------------------------
c
c                  Objective Function Values
c                  *************************
c
c
c
c
c-----------------------------------------------------------------------
      include 'histsmth.inc'
c
c Compute Smooth Sums:
c
      do i=1,nclass
            sval(i) = 0.0
            do is=-nsmooth,nsmooth
                  if(is.ne.0) sval(i) = sval(i) + val(i+is)
            end do
      end do
c
c Objective Function Component:        MEAN  and VARIANCE
c
      if(lmean.or.lvari) then
            rmean = 0.0
            rvar  = 0.0
            do i=1,nclass
                  rmean = rmean + val(i)*zz(i)
                  rvar  = rvar  + val(i)*zzs(i)
            end do
            rm = rmean
            rv = rvar - rmean*rmean
            objmean = abs(rm - mean    )
            objvari = abs(rv - variance)
            write(*,101) rm,mean,rv,variance
 101        format(/' Actual mean     = ',f12.6,' target is ',f12.6,/,
     +             /' Actual variance = ',f12.6,' target is ',f12.6)
      end if
c
c Objective Function Component:        SMOOTHNESS
c
      if(lsmth) then
            objsmth = 0.0
            do i=1,nclass
                  objsmth = objsmth + (val(i)-sval(i)*smthfac(i))
     +                              * (val(i)-sval(i)*smthfac(i))
            end do
            write(*,103) objsmth
 103        format(/' Measure of smoothness = ',f12.9)
      end if
c
c Objective Function Component:        QUANTILES
c
      if(lquan) then
            objquan = 0.0
            do i=1,nuq
                  qact(i) = 0.0
                  do id=1,uqind(i)
                        qact(i) = qact(i) + val(id)
                  end do
                  objquan = objquan + (   (qact(i)-qval(i))
     +                                  * (qact(i)-qval(i)) )
            end do
            write(*,104) objquan/nuq
 104        format(/' Measure of mismatch for quantiles = ',f12.4,//)
      end if
c
c Finished:
c
      write(*,'()')
      return
      end



      subroutine object(ipert,jpert,del)
c-----------------------------------------------------------------------
c
c                  Update Objective Function Values
c                  ********************************
c
c
c
c
c-----------------------------------------------------------------------
      include 'histsmth.inc'
      real     valtryi,valtryj
c
c New tries:
c
      valtryi = val(ipert) + del
      valtryj = val(jpert) - del
c
c Objective Function Component:        MEAN  and VARIANCE
c
      if(lmean.or.lvari) then
            tmean = rmean - val(ipert)* zz(ipert)
     +                    + valtryi   * zz(ipert)
     +                    - val(jpert)* zz(jpert)
     +                    + valtryj   * zz(jpert)
            tvar  = rvar  - val(ipert)*zzs(ipert)
     +                    + valtryi   *zzs(ipert)
     +                    - val(jpert)*zzs(jpert)
     +                    + valtryj   *zzs(jpert)
            rm = tmean
            rv = tvar - tmean*tmean
            obtmean = abs(rm - mean    )
            obtvari = abs(rv - variance)
      end if
c
c Objective Function Component:        SMOOTHNESS
c
      if(lsmth) then
            obtsmth = objsmth
            do i=-nsmooth,nsmooth
                  if(i.ne.0) then 

                        loc = ipert + i
                        if(loc.ge.1.and.loc.le.nclass) then
                           svali(i) = sval(loc) + del
                           obtsmth = obtsmth 
     +                             - (val(loc)-sval(loc)*smthfac(loc))
     +                             * (val(loc)-sval(loc)*smthfac(loc))
     +                             + (val(loc)-svali(i) *smthfac(loc))
     +                             * (val(loc)-svali(i) *smthfac(loc))
                        end if

                        loc = jpert + i
                        if(loc.ge.1.and.loc.le.nclass) then
                           svalj(i) = sval(loc) - del
                           obtsmth = obtsmth 
     +                             - (val(loc)-sval(loc)*smthfac(loc))
     +                             * (val(loc)-sval(loc)*smthfac(loc))
     +                             + (val(loc)-svalj(i) *smthfac(loc))
     +                             * (val(loc)-svalj(i) *smthfac(loc))
                        end if
                  end if
            end do
            obtsmth = obtsmth - (val(ipert)-sval(ipert)*smthfac(ipert))
     +                        * (val(ipert)-sval(ipert)*smthfac(ipert))
     +                        + (valtryi   -sval(ipert)*smthfac(ipert))
     +                        * (valtryi   -sval(ipert)*smthfac(ipert))
            obtsmth = obtsmth - (val(jpert)-sval(jpert)*smthfac(jpert))
     +                        * (val(jpert)-sval(jpert)*smthfac(jpert))
     +                        + (valtryj   -sval(jpert)*smthfac(jpert))
     +                        * (valtryj   -sval(jpert)*smthfac(jpert))
      end if
c
c Objective Function Component:        QUANTILES
c
      if(lquan) then
            do i=1,nuq
                  qtry(i) = qact(i)
            end do
            do i=1,nuq
                  if(ipert.le.uqind(i)) 
     +               qtry(i) = qtry(i) - val(ipert) + valtryi
                  if(jpert.le.uqind(i)) 
     +               qtry(i) = qtry(i) - val(jpert) + valtryj
            end do

            obtquan = 0.0
            do i=1,nuq
                  obtquan = obtquan +     ( (qtry(i)-qval(i))
     +                                     *(qtry(i)-qval(i)) )
            end do
      end if
c
c Finished:
c
      return
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
      open(lun,file='histsmth.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for HISTSMTH',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/data.dat             ',
     +       '-file with data')
      write(lun,12)
 12   format('3   0                        ',
     +       '-   columns for variable and weight')
      write(lun,13)
 13   format('-1.0e21  1.0e21              ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('Primary Variable             ',
     +       '-title')
      write(lun,15)
 15   format('histsmth.ps                  ',
     +       '-file for PostScript output')
      write(lun,16)
 16   format(' 30                          ',
     +       '-   number of hist classes (for display)')
      write(lun,17)
 17   format('histsmth.out                 ',
     +       '-file for smoothed output')
      write(lun,18)
 18   format('100    0.01  100.0           ',
     +       '-smoothing limits: number, min, max')
      write(lun,19)
 19   format('1                            ',
     +       '-0=arithmetic, 1=log scaling')
      write(lun,20)
 20   format('750    50   0.0001   69069   ',
     +       '-maxpert, reporting interval,min Obj,Seed')
      write(lun,21)
 21   format('1  1  1  1                   ',
     +       '-1=on,0=off: mean,var,smth,quantiles')
      write(lun,22)
 22   format('1. 1. 2. 2.                  ',
     +       '-weighting : mean,var,smth,quantiles')
      write(lun,23)
 23   format('  5                          ',
     +       '-size of smoothing window')
      write(lun,24)
 24   format('-999.0 -999.0                ',
     +       '-target mean and variance (-999=>data)')
      write(lun,25)
 25   format('5                            ',
     +       '-number of quantiles: defined from data')
      write(lun,26)
 26   format('0                            ',
     +       '-number of quantiles: defined by user')
      write(lun,27)
 27   format('0.5    1.66                  ',
     +       '-   cdf, z')

      close(lun)
      return
      end
