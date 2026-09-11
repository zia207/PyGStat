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
c                     Bivariate Histogram Plot
c                     ************************
c
c This program generates a PostScript file with a bivariate plot and
c marginal histograms
c
c INPUT/OUTPUT Parameters:
c
c   datafl      file with data
c   ix,iy,iv    columns for the x variable, y variable, and value
c   ilogx,ilogy log scaling for x variable and y variable (1=yes)
c   tmin,tmax   trimming limits
c   univxfl     file with smoothed X distribution
c   ivr,iwt     columns for the variable and weight
c   univyfl     file with smoothed Y distribution
c   ivr,iwt     columns for the variable and weight
c   bivfl       file with smoothed bivariate distribution
c   ivrx,ivry,iwt columns for the variables and weight
c   outfl       file for PostScript output
c   nx,xmn,xmx  X: number, minimum, and maximum
c   ny,ymn,ymx  Y: number, minimum, and maximum
c   imap,icol   color scale map? (0=no, 1=yes), 0=B&W, 1=color
c   pmin,pmax,pinc   limits and labelling for probabilities
c   ncls        number of histogram classes (marginals)
c   title       Title
c
c
c
c The following Parameters control static dimensioning:
c
c   MAXCLS    maximum number of histogram classes
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MAXCLS = 500, VERSION=3.000)
c
c Needed variable declaration:
c
      character  datafl*512,outfl*512,bhistfl*512,uhistflx*512,
     +           uhistfly*512,title*40,xlab*24,ylab*24,str*512,hexa*2
      real       var(20),bhist(MAXCLS,MAXCLS),uhistx(MAXCLS),
     +           uhisty(MAXCLS)
      integer    lin,lpsout,ninx(MAXCLS),niny(MAXCLS), test
      logical    testfl,testdt,testbh,testux,testuy
c
c Dynamic allocation of arrays.
c
      real, allocatable :: xval(:),yval(:),pval(:)
      character*2,allocatable :: iline(:),rline(:),gline(:),bline(:)
c
c For color scale:
c
      common /color/  cmin,cmax,cint(4),cscl
c
c Common variables for Postscript Plotting:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      data    pscl/0.24/,pxmin/0./,pxmax/288./,pymin/0./,
     +        pymax/216./,xmin/0./,xmax/80./,ymin/0./,ymax/60./
c
c Initial Values:
c
      data    lin/1/,lpsout/1/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' BIVPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'bivplt.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'bivplt.par          ') then
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

      read(lin,*,err=98) ivrx,ivry,iwt
      write(*,*) ' column: vx,vy,wt = ',ivrx,ivry,iwt

      read(lin,*,err=98) ilogx,ilogy
      write(*,*) ' log scaling = ',ilogx,ilogy

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=98) uhistflx
      call chknam(uhistflx,512)
      write(*,*) ' X univariate file = ',uhistflx(1:40)

      read(lin,*,err=98) iux,iuxwt
      write(*,*) ' column: vr,wt = ',iux,iuxwt

      read(lin,'(a512)',err=98) uhistfly
      call chknam(uhistfly,512)
      write(*,*) ' Y univariate file = ',uhistfly(1:40)

      read(lin,*,err=98) iuy,iuywt
      write(*,*) ' column: vr,wt = ',iuy,iuywt

      read(lin,'(a512)',err=98) bhistfl
      call chknam(bhistfl,512)
      write(*,*) ' bivariate hist file = ',bhistfl(1:40)

      read(lin,*,err=98) ibx,iby,ibwt
      write(*,*) ' columns: vrx,vry,wt = ',ibx,iby,ibwt

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) imap,icolor
      write(*,*) ' gray/color scale map? gray/color? = ',imap,icolor

      read(lin,*,err=98) pmin,pmax,pinc
      write(*,*) ' pmin,pmax,pinc: ',pmin,pmax,pinc

      read(lin,*,err=98) ncls    
      write(*,*) ' ncls = ',ncls

      read(lin,'(a40)',err=98) title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      close(lin)
      do i=1,ncls
            ninx(i) = 0.0
            niny(i) = 0.0
      end do
c
c Read smoothed univarate histogram for X axis:
c
      inquire(file=uhistflx,exist=testux)
      if(.not.testux) then
            write(*,*) 'ERROR: Must have a smoothed univariate dist.'
            stop
      end if
      open(lin,file=uhistflx,status='OLD')
      read(lin,'()',err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      read(lin,*,err=99) (var(i),i=1,nvari)
      xmin = var(iux)
      backspace lin
      nbhx = 0
 1111 read(lin,*,end=1212,err=99) (var(i),i=1,nvari)
      nbhx = nbhx + 1
      uhistx(nbhx) = var(iuxwt)
      go to 1111
 1212 xmax = var(iux)
      close(lin)
      write(*,*)
      write(*,*) 'Smoothed X distribution: number, min, max ',
     +            nbhx,xmin,xmax
c
c Read smoothed univarate histogram for Y axis:
c
      inquire(file=uhistfly,exist=testuy)
      if(.not.testuy) then
            write(*,*) 'ERROR: Must have a smoothed univariate dist.'
            stop
      end if
      open(lin,file=uhistfly,status='OLD')
      read(lin,'()',err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      read(lin,*,err=99) (var(i),i=1,nvari)
      ymin = var(iuy)
      backspace lin
      nbhy = 0
 1313 read(lin,*,end=1414,err=99) (var(i),i=1,nvari)
      nbhy = nbhy + 1
      uhisty(nbhy) = var(iuywt)
      go to 1313
 1414 ymax = var(iuy)
      close(lin)
      write(*,*)
      write(*,*)'Smoothed Y distribution: number, min, max ',
     +      nbhy,ymin,ymax 
c
c
c Read smoothed bivariate histogram:
c
      inquire(file=bhistfl,exist=testbh)
      if(.not.testbh) then
            write(*,*) 'ERROR: Must have smoothed bivarate dist '
            stop
      end if
      bhmax = 0.0
      open(lin,file=bhistfl,status='OLD')
      read(lin,'()',err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      read(lin,*,err=99) (var(i),i=1,nvari)
      xxx = var(ibx)
      yyy = var(iby)
      backspace lin
      if(abs(xxx-xmin).gt.0.001.or.abs(yyy-ymin).gt.0.001) then
            write(*,*) 'Bivariate histogram may not match marginals:'
            write(*,*) '  xmin, ymin (marginals): ',xmin,ymin
            write(*,*) '  xmin, ymin (bivariate): ',xxx,yyy
      end if
      do iy=1,nbhy
            do ix=1,nbhx
                  read(lin,*,err=99) (var(i),i=1,nvari)
                  xxx          = var(ibx)
                  yyy          = var(iby)
                  bhist(ix,iy) = var(ibwt)
                  if(ix.eq.1) xbhmin = xxx
                  if(ix.eq.2) xbhinc = xxx - xold
                  xold = xxx
                  if(bhist(ix,iy).gt.bhmax) bhmax = bhist(ix,iy)
            end do
            if(iy.eq.1) ybhmin = yyy
            if(iy.eq.2) ybhinc = yyy - yold
            yold = yyy
      end do
      close(lin)
      if(abs(xxx-xmax).gt.0.001.or.abs(yyy-ymax).gt.0.001) then
            write(*,*) 'Bivariate histogram may not match marginals:'
            write(*,*) '  xmax, ymax (marginals): ',xmax,ymax
            write(*,*) '  xmax, ymax (bivariate): ',xxx,yyy
      end if
      hlen = xbhinc/(xmax-xmin)*1000.0
      vlen = xbhinc/(xmax-xmin)*1000.0
c
c Read the data:
c
      write(*,9991)
 9991 format(/' Reading data files.....'/)
      nd = 0
      inquire(file=datafl,exist=testdt)
c
c The first data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      if(testdt) then
            open(lin,file=datafl,status='OLD')
            read(lin,'(a)',err=99) str
            read(lin,*,err=99)     nvari
            do i=1,nvari
                  read(lin,*,err=99)
            end do
            maxdat = 0
 2          read(lin,*,end=4,err=99) (var(j),j=1,nvari)
            if(var(ivrx).lt.tmin.or.var(ivrx).ge.tmax) go to 2
            if(var(ivry).lt.tmin.or.var(ivry).ge.tmax) go to 2
            maxdat = maxdat + 1
            go to 2
 4          continue
c 
c  Allocate the needed memory:
c
            allocate(xval(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 1: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c  
            allocate(yval(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 2: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
c
            allocate(pval(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 3: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
            allocate(iline(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 3: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
            allocate(rline(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 3: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
            allocate(gline(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 3: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
            allocate(bline(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 3: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
c  Now read the data for real:
c
            rewind(lin)
            read(lin,'(a)',err=99) str
            read(lin,*,err=99)     nvari
            do i=1,nvari
                  read(lin,'(a24)',err=99) str(1:24)
                  if(i.eq.ivrx) xlab = str(1:24)
                  if(i.eq.ivry) ylab = str(1:24)
            end do
c
 300        read(lin,*,end=3,err=99) (var(j),j=1,nvari)
            if(var(ivrx).lt.tmin.or.var(ivrx).gt.tmax) go to 300
            if(var(ivry).lt.tmin.or.var(ivry).gt.tmax) go to 300
            if(ilogx.eq.1) then
                  xvalue  = alog10(var(ivrx))
            else 
                  xvalue  = var(ivrx)
            end if 
            if(ilogy.eq.1) then
                  yvalue  = alog10(var(ivry))
            else 
                  yvalue  = var(ivry)
            end if 
            if(iwt.le.0) then
                  weight = 1.0
            else
                  weight = var(iwt)
            end if
            if(xvalue.lt.xmin.or.xvalue.gt.xmax.or.weight.le.0.0.or.
     +         yvalue.lt.ymin.or.yvalue.gt.ymax) go to 300
c
c Accept this data:
c
            nd = nd + 1
            xval(nd) = xvalue
            yval(nd) = yvalue
            pval(nd) = weight
c
c Go back for another data:
c
            go to 300
 3          close(lin)
c
c
            write(*,*) 'Read ',nd,' data '
      end if
c
c Prepare the output file:
c
      write(*,9997)
 9997 format(/' Writing bivariate data.....'/)
      open(unit=lpsout,file=outfl,status='UNKNOWN')
c     
c Add a header:
c
      write(lpsout,998) title(1:20)
c
 998  format('%!PS                                 %    Remove     ',
     +    /, '90 234 translate 1.5 1.5 scale       %  these lines  ',
     +    /, '                                     % for EPSF file ',
     +    /, '%!PS-Adobe-3.0 EPSF-3.0',
     +    /, '%%BoundingBox: 0 0 288 216',
     +    /, '%%Creator: GSLIB',
     +    /, '%%Title:   ',a20,
     +    /, '%%CreationDate: ',
     +    /, '%%EndComments',/,/,/,'%',/,'%',/,'%',/,
     +    /, '/m {moveto} def /l {lineto} def /r {rlineto} def',
     +    /, '/s {stroke} def /n {newpath} def /c {closepath} def',
     +    /, '/rtext{ dup stringwidth pop -1 div 0 rmoveto show } def',
     +    /, '/ctext{ dup stringwidth pop -2 div 0 rmoveto show } def',
     +    /, '/ltext{show} def /gr{grestore} def /gs{gsave} def',
     +    /, '/tr{translate} def /setc{setrgbcolor} def',
     +    /, '/bullet{ 6 0 360 arc c fill } def',/,/,
     +    /, '%72 72 translate',/,/,
     +    /, '0.240000 0.240000 scale')
c
c Customized to bivplt:
c
      write(lpsout,9980) hlen,vlen
 9980 format(//,'/wbullet{ 6 0 360 arc gs 0.8 setgray fill gr} def',
     +    /,    '/hl{n 0 0 m ',f6.3,' 0 l s} bind def',
     +    /,    '/vl{n 0 0 m 0 ',f6.3,' l s} bind def',/,
     +    /,    '0.55 0.55 scale',/)
c
c Draw the color scale map first (if necessary):
c
      if(pmin.ge.pmax) then
            cmin    = 0.0
            cmax    = 0.5 * bhmax
            pinc    = (cmax-cmin) / 10.0
      else
            cmin    = pmin
            cmax    = pmax
            num     = int((cmax-cmin)/pinc+0.5)
            if(num.le.1.or.num.ge.25) pinc    = (cmax-cmin) / 10.0
      end if
      if(imap.eq.1) then
            c0      = 0.00001
            h       = 0.25    * (1.00001*cmax)
            cint(1) = cmin    + c0*cmax
            cint(2) = cint(1) + h
            cint(3) = cint(2) + h
            cint(4) = cint(3) + h
            cscl    = 255*(cint(1)-cmin)/h
            xgmax   = 1000.0
            yinc    = 1000.0 / real(nbhy)
            yloc    = -yinc
            write(lpsout,1000)
 1000       format('575 575 translate')
c
c           Loop over the rows:
c
            do iy=1,nbhy
            yloc = yloc + yinc
c
c Color layers?
c
            
            if(icolor.eq.1) then
c
c           RED, GREEN, and BLUE COLOR LAYERS
c
                  write(lpsout,1001)
 1001             format('/rcl{<')
                  do ix=1,nbhx
                        call red(bhist(ix,iy),iline(ix),rfrac)
                  end do                  
                  write(lpsout,'(200a2)') (iline(i),i=1,nbhx)
                  write(lpsout,1002)
 1002             format('>} def')
                  write(lpsout,1003)
 1003             format('/gcl{<')
                  do ix=1,nbhx
                        call green(bhist(ix,iy),iline(ix),gfrac)
                  end do
                  write(lpsout,'(200a2)') (iline(i),i=1,nbhx)
                  write(lpsout,1002)
                  write(lpsout,1004)
 1004             format('/bcl{<')
                  do ix=1,nbhx
                        call blue(bhist(ix,iy),iline(ix),bfrac)
                  end do
                  write(lpsout,'(200a2)') (iline(i),i=1,nbhx)
                  write(lpsout,1002)
c
c           Draw the COLOR IMAGE on the page:
c
                  write(lpsout,1005) yloc,xgmax,yinc
 1005             format('gsave 0 ',f7.2,' translate ',2f8.2,'  scale')
                  write(lpsout,1006) nbhx,nbhy,nbhx,nbhy
 1006             format(2i4,' 8 [',i3,' 0 0 ',i3,' 0 0] {rcl} {gcl} ',
     +                   ' {bcl} true 3 colorimage grestore')
c
c Black and White?
c
            else
                  do ix=1,nbhx
                        level = 256 - 256*(bhist(ix,iy)-cmin) /
     +                                          (cmax-cmin)
                        level = max(min(level,255),0)
                        iline(ix) = hexa(level)
                  end do
                  write(lpsout,1010)
 1010             format('/gsmap{<')
                  write(lpsout,'(200a2)') (iline(i),i=1,nbhx)
                  write(lpsout,1011) yloc,xgmax,yinc
 1011             format('>} def gsave 0 ',f9.3,' translate ',2f8.2,
     +                   ' scale')
                  write(lpsout,1012) nbhx,nbhy,nbhx,nbhy
 1012             format(2i4,' 8 [',i3,' 0 0 ',i3,' 0 0] {gsmap}',
     +                   ' image grestore')
            end if
c
c           END Loop over rows:
c
            end do
            write(lpsout,1013)
 1013       format('-575 -575 translate')
      end if
     
c
c Draw the cross plot:
c
      call strlen(xlab,24,lostrx)
      call strlen(ylab,24,lostry)
      write(lpsout,101) xmin,xlab(1:lostrx),xmax,
     +                  ymin,ylab(1:lostry),ymax,title
 101  format('%',/,'%',/,'%',/,
     +       '2.5 setlinewidth n  575  575 m 1575  575 l ',/,
     +       '                   1575 1575 l  575 1575 l c s ',/,
     +       '1.0 setlinewidth n  500  575 m  500 1575 l s ',/,
     +       '                 n  575  500 m 1575  500 l s ',/,
     +       '/Helvetica findfont 40 scalefont setfont',/, 
     +       ' 575 515 m  (',f12.3,') ltext',/,
     +       '1000 515 m  (',a,') ctext',/,
     +       '1575 515 m  (',f12.3,') rtext',/,
     +       ' 560 575 translate 90 rotate',/,
     +       '   0 0 m  (',f12.3,') ltext',/,
     +       ' 500 0 m  (',a,') ctext',/,
     +       '1000 0 m  (',f12.3,') rtext',/,
     +       '-90 rotate -560 -575 translate ',/,
     +       '/Helvetica-Bold findfont 50 scalefont setfont',/, 
     +       ' 575 1600 m  (',a40,') ltext',/)
c
c Work out the number of data per class:
c
      do id=1,nd
            ii = 1 + int(((xval(id)-xmin)/(xmax-xmin))*real(ncls))
            ninx(ii) = ninx(ii) + 1
            ii = 1 + int(((yval(id)-ymin)/(ymax-ymin))*real(ncls))
            niny(ii) = niny(ii) + 1
      end do
c
c Note the data points:
c
      do i=1,nd
            xxx = 575.0 + (xval(i)-xmin)/(xmax-xmin)*1000.0
            yyy = 575.0 + (yval(i)-ymin)/(ymax-ymin)*1000.0
            ix = int((xval(i)-xmin)/(xmax-xmin)*nbhx)
            iy = int((yval(i)-ymin)/(ymax-ymin)*nbhy)
            if(icolor.ne.1.and.imap.eq.1) then
                  if(bhist(ix,iy).gt.0.5*cmax) then
                        write(lpsout,502)
                  else
                        write(lpsout,503)
                  end if
            endif
            write(lpsout,504) xxx,yyy
 502        format('1.0 setgray')
 503        format('0.0 setgray')
 504        format('n ',f6.1,1x,f6.1,' bullet')
      end do
c
c Draw the "bottom" X axis histogram Plot:
c
      write(*,9996)
 9996 format(/' Writing histograms.....'/)
      write(lpsout,111)
 111  format('%',/,'% Bottom Histogram Plot:',/,'%',/,'%',/,
     +       '0.0 setgray')
c
c Only show the histogram if there are data:
c
      maxnin = 1.0
      if(nd.gt.0) then
         maxnin = 0.0
         do icls=1,ncls
               if(ninx(icls).gt.maxnin) maxnin = ninx(icls)
               if(niny(icls).gt.maxnin) maxnin = niny(icls)
         end do
         do icls=1,ncls
               zlo = 575.0 + real(icls-1)/real(ncls)      *1000.0
               zup = 575.0 + real(icls  )/real(ncls)      *1000.0
               pup = 500.0 - real(ninx(icls))/real(maxnin)* 500.0
               write(lpsout,112) zlo,zlo,pup,zup,pup,zup
         end do
      end if
 112  format('n ',f6.1,' 500 m ',f6.1,1x,f6.1,' l ',f6.1,1x,f6.1,' l ',
     +     /,'  ',f6.1,' 500 l c gs 0.90 setgray fill gr s')
c
c Draw the smoothed X histogram?
c
      if(testux) then
            sclfac = 500.0 * real(nbhx) / real(ncls) / real(maxnin)
     +                     * max(real(nd),1.0)
            write(lpsout,113)
 113        format('n 575 500 m')
            do i=1,nbhx
                  zz = 575.0 + real(i-1)/real(nbhx) *1000.0
                  pp = 500.0 - uhistx(i) * sclfac
                  write(lpsout,114) zz,pp
            end do
            write(lpsout,115)
      end if
 114  format(f6.1,1x,f6.1,' l')
 115  format('s')
c
c Draw the "side" Y axis histogram Plot:
c
      write(lpsout,121)
 121  format('%',/,'% Side Histogram Plot:',/,'%',/,'%')
c
c Only show the histogram if there are data:
c
      if(nd.gt.0) then
         do icls=1,ncls
               zlo = 575.0 + real(icls-1)/real(ncls)      *1000.0
               zup = 575.0 + real(icls  )/real(ncls)      *1000.0
               pup = 500.0 - real(niny(icls))/real(maxnin)* 500.0
               write(lpsout,122) zlo,pup,zlo,pup,zup,zup
         end do
      end if
 122  format('n 500 ',f6.1,' m ',f6.1,1x,f6.1,' l ',f6.1,1x,f6.1,' l ',
     +     /,' 500 ',f6.1,' l c gs 0.90 setgray fill gr s')
c
c Draw the smoothed Y histogram?
c
      if(testuy) then
            sclfac = 500.0 * real(nbhy) / real(ncls) / real(maxnin)
     +                     * max(real(nd),1.0)
            write(lpsout,123)
 123        format('n 500 575 m')
            do i=1,nbhy
                  zz = 575.0 + real(i-1)/real(nbhy) *1000.0
                  pp = 500.0 - uhisty(i) * sclfac
                  write(lpsout,114) pp,zz
            end do
            write(lpsout,115)
      end if
c
c DRAW A LEGEND BAR:
c
      write(lpsout,124)
 124  format('%',/,'% START OF LEGEND: ',/,'%')
c
c Specification of COLOR IMAGE for scale bar:
c
      num   = 100
      ginc  = (cmax-cmin)/real(num)
      value = cmin - ginc
      do ih=1,num
            value = value + ginc
            if(icolor.eq.0) then
                  level = 256 - 256*(value-cmin)/(cmax-cmin)
                  if(level.lt.0)   level = 0
                  if(level.gt.256) level = 256
                  iline(ih) = hexa(level)
            else
                  if(value.lt.cmin) value = cmin
                  if(value.gt.cmax) value = cmax
                  call red  (value,rline(ih),rfrac)
                  call green(value,gline(ih),gfrac)
                  call blue (value,bline(ih),bfrac)
            end if
      end do
c
c Location of the Color Legend Bar.  Fix the X location and try
c to make the Y location about 80% of the height of the Y dimension:
c
      xtrans = 1600.0
      hscale =   90.0
      ytrans =  675.0
      vscale =  800.0
c
c Only make a legend bar if a color or gray scale map was used:
c
      if(imap.eq.1) then
c
c Place the color / gray scale bar:
c
      if(icolor.eq.0) then
            write(lpsout,201)
 201        format('/gsmap{<')
            write(lpsout,'(200a2)') (iline(i),i=1,num)
            write(lpsout,102) xtrans,ytrans,hscale,vscale
 102        format('>} def gsave ',2f9.3,' translate ',2f8.2,' scale')
            write(lpsout,103) 1,num,1,num
 103        format(2i4,' 8 [',i3,' 0 0 ',i3,' 0 0] {gsmap}',
     +             ' image grestore')
      else
            write(lpsout,205)
            write(lpsout,'(200a2)') (rline(i),i=1,num)
            write(lpsout,206)
            write(lpsout,207)
            write(lpsout,'(200a2)') (gline(i),i=1,num)
            write(lpsout,206)
            write(lpsout,208)
            write(lpsout,'(200a2)') (bline(i),i=1,num)
            write(lpsout,206)
            write(lpsout,209) xtrans,ytrans,hscale,vscale
            write(lpsout,210) 1,num,1,num
 205        format('/rcl{<')
 206        format('>} def')
 207        format('/gcl{<')
 208        format('/bcl{<')
 209        format('gsave ',2f9.3,' translate ',2f8.2,'  scale')
 210        format(2i4,' 8 [',i3,' 0 0 ',i3,' 0 0] {rcl} {gcl} ',
     +             '{bcl} true 3 colorimage grestore')
      endif
c
c Line around the legend:
c
      write(lpsout,125) xtrans,ytrans,xtrans,ytrans+vscale,
     +                  xtrans+hscale,ytrans+vscale,xtrans+hscale,ytrans
 125  format('%',/,'% BORDER AROUND LEGEND: ',/,'%',/,
     +       '0 0 0 setrgbcolor',/,'n ',2f7.1,' m ',2f7.1,' l ',/,
     +       2f7.1,' l ',2f7.1,' l c s',/,/,
     +       '/Helvetica findfont 30 scalefont setfont',/)
c
c Label the legend:
c
      num  = int((cmax-cmin)/pinc+0.5)
      yinc = vscale / real(num)
      yyy  = ytrans - yinc
      xxx  = xtrans + hscale
      cval = cmin - pinc
      do i=1,num+1
            cval = cval + pinc
            yyy  = yyy  + yinc
            call numtext(cval,title(1:12))
            write(lpsout,126) xxx+25.0,yyy,title(1:12)
            write(lpsout,127) xxx,yyy,xxx+15.0,yyy
      end do
 126  format('n ',2f7.1,' m (',a12,') ltext')
 127  format('n ',2f7.1,' m ',2f7.1,' l s')
c
c End of Legend Bar:
c
      end if
c
c Add a footer to the Postscript plot file:
c
      write(lpsout,999)
 999  format('%END OF POSTSCRIPT FILE',/,'4.166667 4.166667 scale',/,/,
     +       '%%EOF',/,'showpage')
c
c Finished:
c
      close(lpsout)
      write(*,9998) VERSION
 9998 format(/' BIVPLT Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'ERROR in parameters somewhere'
 99   stop 'ERROR in data somewhere'
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
      open(lun,file='bivplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for BIVPLT',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('data.dat                  ',
     +       '-file with data')
      write(lun,12)
 12   format('3   5   0                 ',
     +       '-   columns for X variable, Y variable, and wt')
      write(lun,13)
 13   format('1   1                     ',
     +       '-   log scaling for X and Y (0=no, 1=yes)')
      write(lun,14)
 14   format('-0.1   1.0e21             ',
     +       '-   trimming limits')
      write(lun,15)
 15   format('histsmth.sx               ',
     +       '-file with smoothed X distribution')
      write(lun,16)
 16   format('1   2                     ',
     +       '-  columns for variable, weight')
      write(lun,17)
 17   format('histsmth.sy               ',
     +       '-file with smoothed Y distribution')
      write(lun,18)
 18   format('1   2                     ',
     +       '-  columns for variable, weight')
      write(lun,19)
 19   format('scatsmth.out              ',
     +       '-file with smoothed bivariate distribution')
      write(lun,20)
 20   format('1   2   3                 ',
     +       '-  columns for X, Y, and weight')
      write(lun,21)
 21   format('bivplt.ps                 ',
     +       '-file for PostScript output')
      write(lun,22)
 22   format('1      1                  ',
     +       '-pixel map? (0=no, 1=yes), (0=B&W, 1=color)')
      write(lun,23)
 23   format('0.0    0.001  0.0001      ',
     +       '-   probability min, max, and labeling increm')
      write(lun,24)
 24   format('30                        ',
     +       '-number of histogram classes (marginals)')
      write(lun,25)
 25   format('Porosity-Permeability Cross Plot    ',
     +       '-Title')

      close(lun)
      return
      end
