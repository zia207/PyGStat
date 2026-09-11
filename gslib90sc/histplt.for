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
c                           Histogram Plot
c                           **************
c
c This program generates a PostScript file with a histogram and summary
c statistics.
c
c INPUT/OUTPUT Parameters:
c
c   datafl      the input data
c   ivr,iwt     columns for the variable and the weight (0 if non)
c   tmin,tmax   trimming limts (acceptable: >= tmin and < tmax)
c   outfl       output file with histogram
c   hmin,hmax   plotting limts (will choose automatically if hmax<hmin)
c   fmax        frequency plotting maximum
c   ncl         the number of classes
c   ilog        1=log scale, 0=arithmetic
c   icum        1=cumulative histogram, 0=frequency
c   ncum        number of points for cum hist (0 => automatic)
c   ndec        number of decimal points (0 => automatic)
c   title       title for output PostScript file
c   spos        stats position (along X axis)
c   xref        reference value for box plot
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter 
c    file is described in the documentation (see the example hist.par).
c
c 2. If data are <tmin , >=tmax , or the weight is <EPSLON the data will
c    not be considered.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MV=500, MAXCLS=502, EPSLON=1.0e-20, VERSION=3.000)

      character  datafl*512,outfl*512,title*40,varlab*24,str*512,
     +           labfmt*8
      real       var(MV),xh(5),yh(5)
      integer    nincls(MAXCLS),test
      logical    testfl,labnum,reghist,cumhist,connum
c
c Dynamic allocation of arrays that depend on the number of data:
c
      real, allocatable :: ar1(:),ar2(:),ar3(:),ar4(:)

      data lin/1/,hpxmin/1.0/,hpxmax/59.5/,hpymin/0.0/,hpymax/58.0/

      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,wxmin,
     +                wxmax,wymin,wymax

      data lpsout/1/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +     pymin/0.0/,pymax/216.0/,wxmin/-10.0/,wxmax/60.0/,
     +     wymin/-10.0/,wymax/60.0/

c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' HISTPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'histplt.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'histplt.par         ') then
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

      read(lin,*,err=98) ivr,iwt
      write(*,*) ' columns = ',ivr,iwt

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming  limits = ',tmin,tmax

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) hmin,hmax
      write(*,*) ' attribute limits = ',hmin,hmax

      read(lin,*,err=98) fmax
      write(*,*) ' frequency  limit = ',fmax

      labnum = .false.
      read(lin,*,err=98) ncl
      if(ncl.lt.1) labnum = .true.
      ncl = abs(ncl)
      write(*,*) ' number of classes = ',ncl
     
      if((ncl+2).gt.MAXCLS) then
            write(*,*) 'ERROR: exceeded available number of classes'
            write(*,*) '       have ',MAXCLS-2,' available'
            stop
      endif

      read(lin,*,err=98) ilog
      write(*,*) ' log scale option = ',ilog

      reghist = .true.
      cumhist = .false.
      connum  = .false.
      read(lin,*,err=97) i
      write(*,*) ' cumulative frequency option = ',i
      if(i.ne.0) reghist = .false.
      if(i.eq.1) cumhist = .true.
      if(i.eq.2) connum  = .true.
      if(.not.cumhist.and..not.connum) reghist = .true.

      read(lin,*,err=98) ncum
      if(cumhist) write(*,*) ' number of cumulative points = ',ncum
      if(connum ) write(*,*) ' number of points in each class = ',ncum

      read(lin,*,err=98) ndec
      if(ndec.lt.0) ndec = 4
      write(*,*) ' number of decimal places = ',ndec
      write(labfmt,120) ndec
 120  format('(f18.',i1,') ')

      read(lin,'(a40)',err=98) title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      spos = 1.0
      xref = tmin - 1.0
      read(lin,*,end=97,err=98) spos
      write(*,*) ' position of stats = ',spos
      if(spos.lt.-1.5.or.spos.gt.1.5) spos = 1.0

      read(lin,*,end=97,err=98) xref
      write(*,*) ' reference value = ',xref

 97   continue
      close(lin)
c
c Check to make sure the data file exists, then either read in the
c data or write an error message and stop:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      endif
c
c The data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0
 2    read(lin,*,end=4,err=97) (var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax) go to 2
      maxdat = maxdat + 1
      go to 2
 4    continue
c
c Allocate the needed memory:
c
      if(maxdat.lt.1000) maxdat = 1000
      allocate (ar1(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
c     
      allocate (ar2(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
c
      allocate (ar3(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
c
      allocate (ar4(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                       'insufficient memory!', test
            stop
            end if
c
c
c Now, read the data for real:
c
      rewind(lin)
      read(lin,'(a)',err=99) str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'(a24)',err=99) str(1:24)
            if(i.eq.ivr) varlab = str(1:24)
      end do
      if(ivr.gt.nvari) then
            write(*,*) ' ivr is greater than the number in data file!'
            stop
      end if
c
c Read as much data as possible:
c
      vrmin = 1.0e21
      vrmax =-1.0e21
      nd = 0
      nt = 0
 7    read(lin,*,end=8,err=99) (var(j),j=1,nvari)
c
c Trim this data?
c
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax) then
            nt = nt + 1
            go to 7
      endif
      if(iwt.ge.1) then
            if(var(iwt).le.EPSLON) then
                  nt = nt + 1
                  go to 7
            endif
      endif
c
c Accept this data:
c
      nd = nd + 1
      ar1(nd) = var(ivr)
      vrmin = min(ar1(nd),vrmin)
      vrmax = max(ar1(nd),vrmax)
      if(iwt.ge.1) then
            ar2(nd) = var(iwt)
      else
            ar2(nd) = 1.0
      endif
c
c Go back for another data:
c
      go to 7
 8    close(lin)
      if(nd.le.1) then
            write(*,*) 'ERROR: there is less than one datum ',nd
            write(*,*) '       check your data file'
            stop
      endif
c
c Now, rewind the data file and store the data in the arrays.
c


c
c Assign the defaults if necessary:
c
      if(hmax.le.hmin) then
            hmin = vrmin
            hmax = vrmax
      endif
      if(iwt.ge.1) then
            iwt = 1
      else
            iwt = 0
      endif
c
c Open the output file for the PostScript code:
c
      open(lpsout,file=outfl,status='UNKNOWN')
c
c Add a header:
c
      write(lpsout,998) title(1:20)
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
c Get mean and total weight:
c
      xtwt = 0.0
      xmen = 0.0
      do i=1,nd
            xmen = xmen + ar1(i)*ar2(i)
            xtwt = xtwt + ar2(i)
      end do
      xtwti = 1.0  / xtwt
      xmen  = xmen * xtwti
c
c Get the variance:
c
      xvar = 0.0
      do i=1,nd
            xvar = xvar + (ar1(i)-xmen) * (ar1(i)-xmen) * ar2(i)
      end do
      xvar  = xvar * xtwti
c
c Infer some of the histogram parameters:
c
      if(ncl.lt.0) then
            ncl             = 100
            if(nd.le.500) ncl= 50
            if(nd.lt.200) ncl= 20
      endif
      ncl = min((MAXCLS-2),(ncl+2))
      dcl = (hmax-hmin)/real(ncl-2)
      if(ilog.eq.1) then
            if(hmin.le.0.0) hmin = vrmin
            if(hmax.le.0.0) hmax = vrmax
            hmin = real(int(alog10(max(hmin,EPSLON))-0.9))
            hmax = real(int(alog10(max(hmax,EPSLON))+0.9))
      endif
      dcl  = (hmax-hmin)/real(ncl-2)
c
c Determine the histogram class structure:
c
      do i=1,ncl
            ar3(i)    = 0.0
            nincls(i) = 0
      end do
      do i=1,nd
            if(ilog.eq.1) then
                  art = alog10(max(ar1(i),EPSLON))
            else
                  art = ar1(i)
            endif
            wt1 = ar2(i) * xtwti
            j = ((art-hmin)/dcl)+2
            if(j.lt.1)   j = 1
            if(j.gt.ncl) j = ncl
            ar3(j)    =ar3(j)    + wt1
            nincls(j) =nincls(j) + 1
      end do
c
c Sort the Data in Ascending Order:
c
      call sortem(1,nd,ar1,1,ar2,c,d,e,f,g,h)
c
c Turn the weights into a cdf:
c
      oldcp = 0.0
      cp    = 0.0
      do i=1,nd
            cp     = cp + ar2(i) * xtwti
            ar2(i) =(cp + oldcp) * 0.5
            oldcp  = cp
      end do
c
c Obtain the quantiles:
c
      call locate(ar2,nd,1,nd,0.025,i)
      if(i.eq.0) then
            xpt025 = ar1(1)
      else if(i.eq.nd) then
            xpt025 = ar1(nd)
      else
            xpt025 = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.025-ar2(i))/(ar2(i+1)-ar2(i))
      endif
      call locate(ar2,nd,1,nd,0.25,i)
      if(i.eq.0) then
            xlqt = ar1(1)
      else if(i.eq.nd) then
            xlqt = ar1(nd)
      else
            xlqt = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.25-ar2(i))/(ar2(i+1)-ar2(i))
      endif
      call locate(ar2,nd,1,nd,0.50,i)
      if(i.eq.0) then
            xmed = ar1(1)
      else if(i.eq.nd) then
            xmed = ar1(nd)
      else
            xmed = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.50-ar2(i))/(ar2(i+1)-ar2(i))
      endif
      call locate(ar2,nd,1,nd,0.75,i)
      if(i.eq.0) then
            xuqt = ar1(1)
      else if(i.eq.nd) then
            xuqt = ar1(nd)
      else
            xuqt = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.75-ar2(i))/(ar2(i+1)-ar2(i))
      endif
      call locate(ar2,nd,1,nd,0.975,i)
      if(i.eq.0) then
            xpt975 = ar1(1)
      else if(i.eq.nd) then
            xpt975 = ar1(nd)
      else
            xpt975 = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.975-ar2(i))/(ar2(i+1)-ar2(i))
      endif
c
c The coeficient of variation:
c
      xmin = ar1(1)
      xmax = ar1(nd)
      if(xmin.lt.0.0.or.xmen.le.EPSLON) then
            xcvr = -1.0
      else
            xcvr = sqrt(max(xvar,0.0))/xmen
      endif
c
c Write Some of the Statistics to the screen:
c
      write(*,900) nd,xmen,xmed,sqrt(max(xvar,0.0)),xmin,xmax
 900  format(/' There are ',i8,' data with:',/,
     +        '   mean value          = ',f12.5,/,
     +        '   median value        = ',f12.5,/,
     +        '   standard deviation  = ',f12.5,/,
     +        '   minimum and maximum = ',2f12.5,/)
c
c Find the Maximum Class Frequency:
c
      xfrmx  = ar3(1)
      do i=2,ncl
            xfrmx  = max(xfrmx,ar3(i))
      end do
c
c Set to user specified value?
c
      if(fmax.gt.0.0) xfrmx = fmax
c
c Increase xfrmx if labels are added to histogram bars:
c
      if(labnum) xfrmx = xfrmx * 1.15
c
c Change things if we are considering a cumulative histogram:
c
      if(cumhist) then
            xfrmx = 1.0
            do i=2,ncl
                  ar3(i) = ar3(i-1) + ar3(i)
            end do
      end if
c
c Consider a constant number of data in each class?
c
      if(connum) then
            sumwt  = 0.0
            widmin = (hmax-hmin) / 5000.0
            ncl    = 0
            nloc   = 1
            ar2(1) = ar1(1)
 314        ncl  = ncl + 1
            nloc = nloc + ncum
            if(nloc.ge.nd) then
                  ar2(ncl+1) = ar1(nd)
                  ar3(ncl)   = real(ncum-nloc+nd) 
     +                       / max(widmin,(ar2(ncl+1)-ar2(ncl)))
                  sumwt      = sumwt + ar3(ncl)
                  go to 315
            end if
            ar2(ncl+1) = ar1(nloc)
            ar3(ncl)   = real(ncum) / max(widmin,(ar2(ncl+1)-ar2(ncl)))
            sumwt      = sumwt + ar3(ncl)
            go to 314
 315        continue
            xfrmx = 0.0
            do i=1,ncl
                  ar3(i) = ar3(i) / sumwt
                  if(ar3(i).gt.xfrmx) xfrmx = ar3(i)
            end do
      end if
c
c Set some scaling parameters:
c
      xhsmin = hmin
      xhsmax = hmin + real(ncl+1)*dcl
      if(connum) xhsmax = hmax
      yhsmin = 0.0
      yhsmax = 1.03*xfrmx
      xrange = hpxmax - hpxmin
      yrange = hpymax - hpymin
c
c Write the title and the boxes:
c
      ts   = 7.5
      xloc = hpxmin - 0.15*xrange
      yloc = hpymin + 0.50*yrange
      if(cumhist) then
            call pstext(xloc,yloc,20,'Cumulative Frequency',ts,1,90.0,1)
      else
            call pstext(xloc,yloc,9,'Frequency',ts,1,90.0,1)
      end if
c
c Only write the X axis label if a box plot is NOT going to be written:
c
      if(xref.lt.tmin.or.xref.ge.tmax) then
            xloc = hpxmin + 0.50*xrange
            yloc = hpymin - 0.15*yrange
            call pstext(xloc,yloc,24,varlab,ts,1,0.0,1)
      end if
c
c Scale and Draw the histogram:
c
      if(ilog.eq.1) then
            xhsmn = 10.0**xhsmin
            xhsmx = 10.0**xhsmax
      else
            xhsmn = xhsmin
            xhsmx = xhsmax
      end if
      call scal(xhsmn,xhsmx,yhsmin,yhsmax,hpxmin,hpxmax,hpymin,
     +          hpymax,ilog,0)
c
c Regular histogram:
c
      if(reghist) then
            x1 = hmin-dcl
            x2 = hmin
            nh = 5
            do i=1,ncl
                  xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,x1)
                  xh(2) = xh(1)
                  xh(3) = resc(xhsmin,xhsmax,hpxmin,hpxmax,x2)
                  xh(4) = xh(3)
                  xh(5) = xh(1)
                  yh(1) = hpymin
                  yh(2) = resc(yhsmin,yhsmax,hpymin,hpymax,ar3(i))
                  if(yh(2).gt.hpymax) yh(2) = hpymax
                  yh(3) = yh(2)
                  yh(4) = yh(1)
                  yh(5) = yh(1)
                  if(ar3(i).gt.xfrmx) then
                        call psfill(nh,xh,yh,0.3,0.0)
                  else
                        call psfill(nh,xh,yh,0.3,0.9)
                  end if
c
c Label number of data in class:
c
                  if(labnum.and.nincls(i).gt.0) then
                        write(varlab,333) nincls(i)
 333                    format('  ',i10)
                        xloc = xh(3)
                        yloc = yh(2)
                        call pstext(xloc,yloc,12,varlab,6.0,1,90.0,0)
                  end if
c
c End Label number of data in class
c
                  x1=x1+dcl
                  x2=x2+dcl
            end do
      end if
c
c Cumulative Histogram:
c
      if(cumhist) then
            nloop = nd
            if(ncum.gt.1.and.ncum.lt.nd) nloop = ncum
            do i=1,nloop
                  pp   = (real(i)-0.5)/real(nloop)
                  call locate(ar2,nd,1,nd,pp,j)
                  if(j.eq.0) then
                        xloc = ar1(1)
                  else if(j.eq.nd) then
                        xloc = ar1(nd)
                  else
                        xloc = ar1(j) +      (ar1(j+1)-ar1(j)) *
     +                                  (pp-ar2(j))/(ar2(j+1)-ar2(j))
                  end if
                  if(ilog.eq.1) xloc = alog10(xloc+EPSLON)
                  ar3(i) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xloc)
                  if(ar3(i).lt.hpxmin) ar3(i) = hpxmin
                  if(ar3(i).gt.hpxmax) ar3(i) = hpxmax
                  ar4(i) = resc(yhsmin,yhsmax,hpymin,hpymax,pp)
                  if(ar4(i).lt.hpymin) ar4(i) = hpymin
                  if(ar4(i).gt.hpymax) ar4(i) = hpymax
            end do
            call psline(nloop,ar3,ar4,0.5,0)
            ar1(1) = hpxmin
            ar1(2) = hpxmax
            ar2(1) = resc(yhsmin,yhsmax,hpymin,hpymax,1.0)
            ar2(2) = ar2(1)
            call psline(2,ar1,ar2,0.5,0)
      end if
c
c Histogram with constant number of data:
c
      if(connum) then
            nh = 5
            do i=1,ncl
                  x1 = ar2(i)
                  x2 = ar2(i+1)
                  xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,x1)
                  xh(2) = xh(1)
                  xh(3) = resc(xhsmin,xhsmax,hpxmin,hpxmax,x2)
                  xh(4) = xh(3)
                  xh(5) = xh(1)
                  yh(1) = hpymin
                  yh(2) = resc(yhsmin,yhsmax,hpymin,hpymax,ar3(i))
                  if(yh(2).gt.hpymax) yh(2) = hpymax
                  yh(3) = yh(2)
                  yh(4) = yh(1)
                  yh(5) = yh(1)
                  if(ar3(i).gt.xfrmx) then
                        call psfill(nh,xh,yh,0.3,0.0)
                  else
                        call psfill(nh,xh,yh,0.3,0.9)
                  end if
            end do
      end if
c
c Write the Statistics to the Output Device:
c
      x1 = hpxmin + 0.74*xrange
      x2 = hpxmin + 0.76*xrange
      yd = 0.044*yrange
      yd2= 0.060*yrange
      yy = hpymax
      ts =  7.0
      ls = 40
      yy = hpymax + 0.01*(hpxmax-hpxmin)
      call pstext(hpxmin,yy,40,title,8.0,3,0.0,0)
c
c Position the labels:
c
      xtr = 250.0*spos - 150.0
                  ytr =  -30.0
      if(cumhist) ytr = -300.0
      write(lpsout,201) xtr,ytr
 201  format(/,'gsave ',f7.1,1x,f7.1,' translate',/)
c
c Write the statistics:
c
      str = 'Number of Data'
      call histtext(ls,str,x1,yy,ts,0.0,2)
      write(str,'(i8)') nd
      call histtext(ls,str,x2,yy,ts,yd2,0)
      if(nt.gt.0) then
            str = 'number trimmed'
            call histtext(ls,str,x1,yy,ts,0.0,2)
            write(str,'(i8)') nt
            call histtext(ls,str,x2,yy,ts,yd2,0)
      endif
      str = 'mean'
      call histtext(ls,str,x1,yy,ts,0.0,2)
      write(str,labfmt) xmen
      call histtext(ls,str,x2,yy,ts,yd,0)
      
      str = 'std. dev.'
      call histtext(ls,str,x1,yy,ts,0.0,2)
      write(str,labfmt) sqrt(max(xvar,0.0))
      call histtext(ls,str,x2,yy,ts,yd,0)
      str = 'coef. of var'
      call histtext(ls,str,x1,yy,ts,0.0,2)
      if(xcvr.lt.0.0) then
            str = ' undefined          '
      else
            write(str,labfmt) xcvr
      endif
      call histtext(ls,str,x2,yy,ts,yd2,0)
      str = 'maximum'
      call histtext(ls,str,x1,yy,ts,0.,2)
      write(str,labfmt) xmax
      call histtext(ls,str,x2,yy,ts,yd,0)
      str = 'upper quartile'
      call histtext(ls,str,x1,yy,ts,0.,2)
      write(str,labfmt) xuqt
      call histtext(ls,str,x2,yy,ts,yd,0)
      str = 'median'
      call histtext(ls,str,x1,yy,ts,0.,2)
      write(str,labfmt) xmed
      call histtext(ls,str,x2,yy,ts,yd,0)
      str = 'lower quartile'
      call histtext(ls,str,x1,yy,ts,0.,2)
      write(str,labfmt) xlqt
      call histtext(ls,str,x2,yy,ts,yd,0)
      str = 'minimum'
      call histtext(ls,str,x1,yy,ts,0.,2)
      write(str,labfmt) xmin
      call histtext(ls,str,x2,yy,ts,yd2,0)
      if(iwt.ge.1) then
            str = 'weights used'
            call histtext(ls,str,x1,yy,ts,0.,2)
      end if

      write(lpsout,202)
 202  format(/,'grestore',/)
c
c Write a box plot?
c
      if(xref.ge.tmin.and.xref.lt.tmax) then
            write(lpsout,1001)
1001        format("%START BOX PLOT")
            if(ilog.eq.1) then
                  xpt025 = alog10(xpt025)
                  xlqt   = alog10(xlqt)
                  xmed   = alog10(xmed)
                  xuqt   = alog10(xuqt)
                  xpt975 = alog10(xpt975)
                  if(xref.gt.0.0) xref = alog10(xref)
            end if
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xpt025)
            yh(1) = -8
            xh(2) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xlqt)
            yh(2) = -8
            call psline(2, xh, yh, 0.5, 0)
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xuqt)
            yh(1) = -8
            xh(2) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xpt975)
            yh(2) = -8
            call psline(2, xh, yh, 0.5, 0)
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xlqt)
            yh(1) = -6
            xh(2) = xh(1)
            yh(2) = -10
            xh(3) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xuqt)
            yh(3) = yh(2)
            xh(4) = xh(3)
            yh(4) = yh(1)
            xh(5) = xh(1)
            yh(5) = yh(1)
            call psline(5, xh, yh, 1.0, 0)
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xpt025)
            yh(1) = -11
            xh(2) = xh(1)
            yh(2) = -5
            call psline(2, xh, yh, 1.0, 0)
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xpt975)
            yh(1) = -11
            xh(2) = xh(1)
            yh(2) = -5
            call psline(2, xh, yh, 1.0, 0)
            xh(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xmed)
            yh(1) = -10
            xh(2) = xh(1)
            yh(2) = -6
            call psline(2, xh, yh, 2.0, 0)
            xs = resc(xhsmin,xhsmax,hpxmin,hpxmax,xref)
            xs = max(min(xs,hpxmax),hpxmin)
            ys = -8
            ix = int((resc(wxmin,wxmax,pxmin,pxmax,xs))/pscl)
            iy = int((resc(wymin,wymax,pymin,pymax,ys))/pscl)
            irad = 10
            write(lpsout,1002) ix,iy,irad
1002        format(i4, 1x, i4, 1x, i3, ' 0 360 arc c fill')
            write(lpsout,1003)
1003        format("%END BOX PLOT")
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
 9998 format(/' HISTPLT Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'ERROR in parameters somewhere'
 99   stop 'ERROR in data somewhere'
      end



      subroutine histtext(lostr,str,x,y,ts,yd,iadj)
c-----------------------------------------------------------------------
c
c A space saving routine that either writes the text to the screen or
c writes the text to a postscript file and then decrements the y 
c position for the next time.
c
c INPUT:
c      lostr - length of the string
c      str   - the character string
c      x,y   - the location to position the text
c      ts    - text size
c      yd    - amount top decrement the y position
c      iadj  - adjustment: 0 left, 1 center, 2 right
c
c-----------------------------------------------------------------------
      character str*80
c
c Common variables for Postscript Plotting:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
c
c Plot the text on the requested device:
c
      it = 1
      rt = 0.0
      call pstext(x,y,lostr,str,ts,it,rt,iadj)
      y = y - yd
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
      open(lun,file='histplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for HISTPLT',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat          ',
     +       '-file with data')
      write(lun,12)
 12   format('3   0                        ',
     +       '-   columns for variable and weight')
      write(lun,13)
 13   format('-1.0     1.0e21              ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('histplt.ps                   ',
     +       '-file for PostScript output')
      write(lun,15)
 15   format(' 0.0      20.0               ',
     +       '-attribute minimum and maximum')
      write(lun,16)
 16   format('-1.0                         ',
     +       '-frequency maximum (<0 for automatic)')
      write(lun,17)
 17   format('20                           ',
     +       '-number of classes')
      write(lun,18)
 18   format('0                            ',
     +       '-0=arithmetic, 1=log scaling')
      write(lun,19)
 19   format('0                            ',
     +       '-0=frequency,  1=cumulative histogram')
      write(lun,20)
 20   format('0                            ',
     +       '-   number of cum. quantiles (<0 for all)')
      write(lun,21)
 21   format('2                            ',
     +       '-number of decimal places (<0 for auto.)')
      write(lun,22)
 22   format('Clustered Data               ',
     +       '-title')
      write(lun,23)
 23   format('1.5                          ',
     +       '-positioning of stats (L to R: -1 to 1)')
      write(lun,24)
 24   format('-1.1e21                      ',
     +       '-reference value for box plot')

      close(lun)
      return
      end
