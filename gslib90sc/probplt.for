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
c                  Normal/Lognormal Probability Plot
c                  *********************************
c
c Displays a set of data values on a probability plot with either an
c arithmetic or logarithmic scaling.
c
c INPUT/OUTPUT Parameters:
c
c   datafl         the data file
c   ivr,iwt        columns for the variable and the weight
c   tmin,tmax      trimming limits
c   outfl          output file for PostScript probability plot
c   npts           number of points to plot (negative means plot all)
c   ilog           1=logarithmic scale, 0=arithmetic
c   min, max, inc  min value, max value and increment for labeling
c   title          the title
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter
c    file is described in the documentation (see example scatplt.par)
c
c 2. The logarithmic scaling is base 10
c
c 3. All acceptable values outside the plotting limits (including
c    zero or negative values when a logarithmic scale is used) are
c    used to establish the cdf but are not shown.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MV=500, NPVAL = 21, ZERO=0.0, BIGNUM=1.0e21,
     +          EPSLON=1.0e-21, VERSION=3.000)

      real      var(MV),xx(MV),yy(MV),pval(NPVAL)
      integer   test
      real*8    pdp
      character datafl*512,outfl*512,varlab*24,title*40,str*512
      logical   testfl,plotall
c
c Dynamic allocation of arrays that depend on the number of data.
c
      real, allocatable :: ar1(:), ar2(:)
      
      data lin/1/,lpsout/1/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +     pymin/0.0/,pymax/216.0/,xmin/-10.0/,xmax/60.0/,
     +     ymin/-10.0/,ymax/60.0/,hpxmin/1.0/,hpxmax/59.5/,
     +     hpymin/0.0/,hpymax/58.0/

      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin, 
     +                xmax,ymin,ymax
c
c The probability scaling and labelling:
c
      data pval/0.0001,0.001,0.002,0.01,0.02,0.05,0.10,0.20,0.30,
     +          0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99,0.998,
     +          0.999,0.9999/
c  
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' PROBPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'probplt.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'probplt.par         ') then
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
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=97) ivr,iwt
      write(*,*) ' columns = ',ivr,iwt

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' tmin and tmax = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' outfl = ',outfl(1:40)

      read(lin,*,err=97) npts
      write(*,*) ' number of points to plot = ',npts

      read(lin,*,err=97) ilog
      write(*,*) ' logarithm scale option = ',ilog

      read(lin,*,err=97) pmin,pmax,pinc
      write(*,*) ' min, max, increment = ',pmin,pmax,pinc

      read(lin,'(a40)',err=97) title
      call chktitle(title,40)
      write(*,*) ' title = ',title
      write(*,*)

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
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,*) 
      end do
      maxdat = 0
 20   read(lin,*,end=40,err=99)(var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 20
      maxdat = maxdat + 1
      go to 20
 40   continue
c
c Allocate the needed memory.
c
      allocate (ar1(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (ar2(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
c Now read the data for real.
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
      nd = 0
      nt = 0
      vrmin =  BIGNUM
      vrmax = -BIGNUM
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
      if(nd.lt.1)then
            write(*,*)'Not reading any data.'
            stop
      endif
c
c Get mean and total weight:
c
      xtwt = 0.0
      xmen = 0.0
      do i=1,nd
            xmen = xmen + ar1(i)*ar2(i)
            xtwt = xtwt + ar2(i)
      end do
      if(xtwt.lt.EPSLON) stop 'Cumulative Probability too LOW'
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
c Sort the Data in Ascending Order:
c
      call sortem(1,nd,ar1,1,ar2,c,d,e,f,g,h)
c
c Get cumulative probability and normalize:
c
      oldcp = 0.0
      cp    = 0.0
      do i=1,nd
            cp     = cp + ar2(i)*xtwti
            ar2(i) = 0.5*(cp+oldcp)
            oldcp  = cp
      end do
      call locate(ar2,nd,1,nd,0.50,i)
      if(i.eq.0) then
            xmed = ar1(1)
      else if(i.eq.nd) then
            xmed = ar1(nd)
      else
            xmed = ar1(i) +      (ar1(i+1)-ar1(i)) *
     +                      (0.50-ar2(i))/(ar2(i+1)-ar2(i))
      endif
c
c Write Some of the Statistics to the screen:
c
      write(*,900) nd,xmen,xmed,sqrt(max(xvar,0.0)),vrmin,vrmax,nt
 900  format(/' There are ',i8,' data with:',/,
     +        '   mean value          = ',f12.5,/,
     +        '   median value        = ',f12.5,/,
     +        '   standard deviation  = ',f12.5,/,
     +        '   minimum and maximum = ',2f12.5,/,
     +        '   number trimmed      = ',i8,/)
c
c Plot the probability plot:
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
c Adjust the scaling parameters if undefined or if log scaling is used:
c
      if(pmax.le.pmin) then
            pmin = ar1(1)
            pmax = ar1(nd)
            pinc = (pmax-pmin)/10.0
            write(*,*) 'Estimating limits for the plot:',pmin,pmax
      endif
      if(ilog.eq.1) then
c           pmin   = max(ar1(1), 0.0001)
c           pmax   = max(ar1(nd),0.0001)
            istart = int(alog10(pmin)-0.999)
            start  = 10.0**istart
            iend   = int(alog10(pmax)+0.999)
            ncyc   = iend - istart
            pmin   = real(istart)
            pmax   = real(iend)
      endif
c
c Initialize:
c
      xhsmin = pmin
      xhsmax = pmax
      pdp = dble(pval(1))
      call gauinv(pdp,yhsmin,ierr)
      pdp = dble(pval(npval))
      call gauinv(pdp,yhsmax,ierr)
      xrange = hpxmax - hpxmin
      yrange = hpymax - hpymin
c
c Write the title and the labels:
c
      ts   = 7.5
      xloc = hpxmin - 0.15*xrange
      yloc = hpymin + 0.50*yrange
      call pstext(xloc,yloc,22,'Cumulative Probability',ts,
     +                      1,90.0,1)
      xloc = hpxmin + 0.50*xrange
      yloc = hpymin - 0.15*yrange
      call pstext(xloc,yloc,24,varlab,ts,1,0.0,1)
      yloc = hpymax + 0.01*(hpxmax-hpxmin)
      call pstext(hpxmin,yloc,40,title,9.,3,0.,0)
c
c Draw a border box:
c
      xx(1) = hpxmin
      yy(1) = hpymin
      xx(2) = hpxmin
      yy(2) = hpymax
      xx(3) = hpxmax
      yy(3) = hpymax
      xx(4) = hpxmax
      yy(4) = hpymin
      xx(5) = hpxmin
      yy(5) = hpymin
      np    = 5
      siz   = 0.7
      idsh  = 0
      call psline(np,xx,yy,siz,idsh)
c
c Scale and Draw the Probability Axis:
c
      np    = 2
      siz   = 0.4
      xx(1) = hpxmin
      xx(2) = hpxmax
      xloc  = hpxmin - 0.01*xrange
      ts    = 7.0
      do i=1,npval
            pdp = dble(pval(i))
            call gauinv(pdp,yloc,ierr)
            yy(1) = resc(yhsmin,yhsmax,hpymin,hpymax,yloc)
            yy(2) = yy(1)
            call psline(np,xx,yy,siz,idsh)
            yloc = yy(1) - 0.01*yrange
            cp   = pval(i) * 100.0
            write(title(1:5),'(i5)') int(cp)
            if(i.eq.1.or.i.eq.npval) write(title(1:5),'(f5.2)') cp
            if(i.eq.2.or.i.eq.3.or.i.eq.19.or.i.eq.20)
     +          write(title(1:5),'(f5.1)') cp
            call pstext(xloc,yloc,5,title,ts,1,0.0,2)
      end do
c
c Scale and Draw the Variable Axis:
c
      yy(1) = hpymin
      yy(2) = hpymax
      yloc  = hpymin - 0.05*xrange
      if(ilog.eq.1) then
c
c      LOGARITHMIC SCALE:
c
            start  = start / 10.0
            do i=1,ncyc+1
                  start = start * 10.0
                  xloc = alog10(start)
                  xloc = resc(xhsmin,xhsmax,hpxmin,hpxmax,xloc)
                  a = abs(start)
                  s = start
                           write(title(1:8),'(f8.0)')s
                  if(a.lt.10) write(title(1:8),'(f8.1)')s
                  if(a.lt.1.) write(title(1:8),'(f8.2)')s
                  if(a.lt.0.1) write(title(1:8),'(f8.3)')s
                  if(a.lt.0.01) write(title(1:8),'(f8.4)')s
                  if(a.lt.0.001) write(title(1:8),'(f8.5)')s
                  if(a.lt.0.0001) write(title(1:8),'(f8.6)')s
                  if(a.lt.0.00001) write(title(1:8),'(f8.7)')s
                  call pstext(xloc,yloc,8,title,ts,1,0.0,1)
                  if(i.le.ncyc) then
                        do j=1,9
                              xloc = alog10(start*j)
                              xx(1) = resc(xhsmin,xhsmax,
     +                                          hpxmin,hpxmax,xloc)
                              xx(2) = xx(1)
                              call psline(np,xx,yy,siz,idsh)
                        end do
                  endif
            end do
      else
c
c      ARITHMETIC SCALE:
c
            nloop = int((pmax-pmin)/pinc) + 1
            temp  = pmin-pinc
            do i=1,nloop
                  temp = temp + pinc
                  xloc = resc(xhsmin,xhsmax,hpxmin,hpxmax,temp)
                  s = temp
                  a = abs(temp)
                           write(title(1:8),'(f8.0)')s
                  if(a.lt.10) write(title(1:8),'(f8.1)')s
                  if(a.lt.1.) write(title(1:8),'(f8.2)')s
                  if(a.lt.0.1) write(title(1:8),'(f8.3)')s
                  if(a.lt.0.01) write(title(1:8),'(f8.4)')s
                  if(a.lt.0.001) write(title(1:8),'(f8.5)')s
                  if(a.lt.0.0001) write(title(1:8),'(f8.6)')s
                  if(a.lt.0.00001) write(title(1:8),'(f8.7)')s
                  if(s.eq.0.0) write(title(1:8),'(f8.0)')s
                  call pstext(xloc,yloc,8,title,ts,1,0.0,1)
                  xx(1) = xloc
                  xx(2) = xx(1)
                  call psline(np,xx,yy,siz,idsh)
            end do
      endif
c
c Scale and Draw the points on the plot:
c
      nloop   =  nd
      plotall = .true.
      if(npts.le.0) npts = nd
      if(npts.lt.nd) then
            nloop   =  npts
            plotall = .false.
            write(*,901) nloop
 901        format(' Plotting  ',i8,' equally spaced quantiles',/)
      end if
      do i=1,nloop
            if(plotall) then
                  xloc =      ar1(i)
                  pdp  = dble(ar2(i))
            else
                  pp   = (real(i)-0.5)/real(nloop)
                  pdp  = dble(pp)
                  call locate(ar2,nd,1,nd,pp,j)
                  if(j.eq.0) then
                        xloc = ar1(1)
                  else if(j.eq.nd) then
                        xloc = ar1(nd)
                  else
                        xloc = ar1(j) +      (ar1(j+1)-ar1(j)) *
     +                                  (pp-ar2(j))/(ar2(j+1)-ar2(j))
                  end if
            end if
            if(ilog.ne.0) then
                  if(xloc.le.EPSLON) go to 10
                  xloc  = alog10(xloc)
            endif
            call gauinv(pdp,yloc,ierr)
            if(xloc.lt.xhsmin.or.xloc.gt.xhsmax.or.
     +         yloc.lt.yhsmin.or.yloc.gt.yhsmax) go to 10
            xx(1) = resc(xhsmin,xhsmax,hpxmin,hpxmax,xloc)
            yy(1) = resc(yhsmin,yhsmax,hpymin,hpymax,yloc)
            ix = int((resc(xmin,xmax,pxmin,pxmax,xx(1)))/ pscl)
            iy = int((resc(ymin,ymax,pymin,pymax,yy(1)))/ pscl)
            write(lpsout,101) ix,iy
 101        format('n ',i4.4,1x,i4.4,' bullet')
 10         continue
      end do
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
 9998 format(/' PROBPLT Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in parameters somewhere'
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
      open(lun,file='probplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for PROBPLT',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat            ',
     +       '-file with data')
      write(lun,12)
 12   format('3   5                          ',
     +       '-  columns for variable and weight')
      write(lun,13)
 13   format('-1.0  1.0e21                   ',
     +       '-  trimming limits')
      write(lun,14)
 14   format('probplt.ps                     ',
     +       '-file for PostScript output')
      write(lun,15)
 15   format('0                              ',
     +       '-number of points to plot (<0 for all)')
      write(lun,16)
 16   format('0                              ',
     +       '-0=arithmetic, 1=log scaling')
      write(lun,17)
 17   format('0.0   30.0   5.0               ',
     +       '-min,max,increment for labeling')
      write(lun,18)
 18   format('Clustered Data                 ',
     +       '-title')

      close(lun)
      return
      end
