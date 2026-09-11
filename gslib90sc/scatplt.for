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
c                  Scatterplot/Bivariate Statistics
c                  *********************************
c
c INPUT/OUTPUT Parameters:
c
c   datafl           the input data file
c   icx,icy,iwt,i3   columns for X, Y, weight, third variable
c   tmin,tmax        trimming limits
c   outfl            the output PostScript file
c   pminx,pmaxx      plotting limits on X variable
c   pminy,pmaxy      plotting limits on Y variable
c   nth              plot every nth point
c   bsize            relative size of bullet
c   title            title
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter 
c    file is described in the documentation (see example scatplt.par)
c
c 2. The calculation of the rank correlation coefficient does not
c    handle spikes correctly, nor does it account for declustering
c    weights.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MV=500, EPSLON=1.0e-21, VERSION=3.000)
      integer    test
      character  datafl*512,outfl*512,title*40,str*512,xlab*24,ylab*24
      real       var(MV)
      logical    testfl
c
c Declare dynamic arrays that depend on the number of data:
c
      real, allocatable :: vr1(:), vr2(:), vr3(:), wt(:)
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
c
      data lin/1/,lpsout/2/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +     pymin/0.0/,pymax/216.0/,xmin/-10.0/,xmax/83.3/,
     +     ymin/-10.0/,ymax/60.0/,hpxmin/1.0/,hpxmax/59.0/,
     +     hpymin/0.0/,hpymax/58.0/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' SCATPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'scatplt.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'scatplt.par         ') then
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

      read(lin,*,err=97) ivr1,ivr2,iwt,ivr3
      write(*,*) ' columns = ',ivr1,ivr2,iwt,ivr3

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' X trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) xscmin,xscmax,ilogx
      write(*,*) ' X plotting limits = ',xscmin,xscmax,ilogx

      read(lin,*,err=97) yscmin,yscmax,ilogy
      write(*,*) ' Y plotting limits = ',yscmin,yscmax,ilogy

      read(lin,*,err=97) nth
      if(nth.lt.1) nth = 1
      write(*,*) ' plot subsetting = ',nth

      read(lin,*,err=97) bsize
      if(bsize.le.0.0.or.bsize.gt.10.0) bsize = 1.0
      write(*,*) ' relative bullet size = ',bsize

      read(lin,*,err=97) gmin,gmax
      write(*,*) ' gray scale limits for third variable = ',gmin,gmax

      read(lin,'(a40)',err=97) title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      close(lin)
c
c Check to make sure the data file exists, then either write an error
c message and stop, or read in as much data as possible:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      endif
c
c Read data:
c
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,*,err=99)
      end do
      maxdat = 0
 20   read(lin,*,end=30,err=99) (var(j),j=1,nvari)
      maxdat = maxdat + 1
      go to 20
 30   continue
c       
c Allocate the needed memory.
c
      allocate (vr1(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c      
      allocate (vr2(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c      
      allocate (vr3(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c      
      allocate (wt(maxdat), stat = test)
            if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c      
c      
c Now, read the data into the arrays:
c
      rewind(lin)
      read(lin,'(a)',err=99) str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'(a24)',err=99) str(1:24)
            if(i.eq.ivr1) xlab = str(1:24)
            if(i.eq.ivr2) ylab = str(1:24)
      end do
      if(ivr1.le.0.or.ivr1.gt.nvari.or.ivr2.le.0.or.ivr2.gt.nvari) then
            write(*,*) ' ERROR: ivr is invalid '
            write(*,*) '        >0 and <number of variables in file'
            stop
      end if
      if(iwt.gt.nvari) then
            write(*,*) ' ERROR: iwt is too large '
            stop
      end if
      nd   = 0
      nt   = 0
      xtwt = 0.0
      v1min = 1.0e21
      v1max =-1.0e21
      v2min = 1.0e21
      v2max =-1.0e21
 2    read(lin,*,end=3,err=99) (var(j),j=1,nvari)
c
c Trim this data?
c
      if(var(ivr1).lt.tmin.or.var(ivr1).ge.tmax.or.
     +   var(ivr2).lt.tmin.or.var(ivr2).ge.tmax) then
            nt = nt + 1
            go to 2
      endif
      if(iwt.ge.1) then
            if(var(iwt).le.EPSLON) then
                  nt = nt + 1
                  go to 2
            endif
      endif
c
c Accept this data:
c
      nd = nd + 1
      vr1(nd) = var(ivr1)
      vr2(nd) = var(ivr2)
      if(iwt.ge.1) then
            wt(nd) = var(iwt)
      else
            wt(nd) = 1.0
      endif
      if(ivr3.ge.1) then
            vr3(nd) = var(ivr3)
      else
            vr3(nd) = -1.1e21
      endif
      xtwt = xtwt + wt(nd)
      if(vr1(nd).lt.v1min) v1min = vr1(nd)
      if(vr1(nd).gt.v1max) v1max = vr1(nd)
      if(vr2(nd).lt.v2min) v2min = vr2(nd)
      if(vr2(nd).gt.v2max) v2max = vr2(nd)
c
c Go back for another data:
c
      go to 2
 3    close(lin)
      if(nd.le.1) then
            write(*,*) ' ERROR: too few data ',nd
            stop
      endif
c
c Compute the needed statistics:
c
      xmn  = 1.0e21
      xmx  =-1.0e21
      ymn  = 1.0e21
      ymx  =-1.0e21
      vr1m = 0.0
      vr2m = 0.0
      do i=1,nd
            wt(i)= wt(i) / xtwt
            vrt1 = vr1(i)
            vrt2 = vr2(i)
            xmn  = min(xmn,vrt1)
            xmx  = max(xmx,vrt1)
            ymn  = min(ymn,vrt2)
            ymx  = max(ymx,vrt2)
            vr1m = vr1m + vrt1*wt(i)
            vr2m = vr2m + vrt2*wt(i)
      end do
      vr1v = 0.0
      vr2v = 0.0
      vr12 = 0.0
      do i=1,nd
            vrt1 = vr1(i)
            vrt2 = vr2(i)
            vr1v = vr1v + (vrt1-vr1m)*(vrt1-vr1m)*wt(i)
            vr2v = vr2v + (vrt2-vr2m)*(vrt2-vr2m)*wt(i)
            vr12 = vr12 + (vrt1-vr1m)*(vrt2-vr2m)*wt(i)
      end do
      corr = vr12 / sqrt(max((vr2v*vr1v),0.0))
c
c Write Some of the Statistics to the screen:
c
      write(*,900) nd,vr1m,vr2m,sqrt(max(vr1v,0.0)),
     +             sqrt(max(vr2v,0.0)),corr,xmn,xmx,ymn,ymx
 900  format(/' There are ',i8,' data with:',/,
     +        '   X mean value         = ',f12.5,/,
     +        '   Y mean value         = ',f12.5,/,
     +        '   X standard deviation = ',f12.5,/,
     +        '   Y standard deviation = ',f12.5,/,
     +        '   Correlation Coeff.   = ',f12.5,/,
     +        '   X min and max        = ',2f12.5,/,
     +        '   Y min and max        = ',2f12.5,/)
c
c Assign the defaults if necessary:
c
      if(xscmax.le.xscmin) then
            xscmin = xmn
            xscmax = xmx
      endif
      if(yscmax.le.yscmin) then
            yscmin = ymn
            yscmax = ymx
      endif
c
c Log scaling?
c
      if(ilogx.eq.1) then
            if(xscmin.le.0.0) xscmin = max(v1min,0.00001)
            if(xscmax.le.0.0) xscmax = v1max
            xscmin = real(int(alog10(max(xscmin,EPSLON))-0.9))
            xscmax = real(int(alog10(max(xscmax,EPSLON))+0.9))
      endif
      if(ilogy.eq.1) then
            if(yscmin.le.0.0) yscmin = max(v2min,0.00001)
            if(yscmax.le.0.0) yscmax = v2max
            yscmin = real(int(alog10(max(yscmin,EPSLON))-0.9))
            yscmax = real(int(alog10(max(yscmax,EPSLON))+0.9))
      endif
      xrange = hpxmax - hpxmin
      yrange = hpymax - hpymin
c
c Open the output file and add a header:
c
      bsize = 6.0*bsize
      open(lpsout,file=outfl,status='UNKNOWN')
      write(lpsout,998) title(1:20),bsize
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
     +    /, '/bullet{ ',f5.2,' 0 360 arc c fill } def',/,/,
     +    /, '%72 72 translate',/,/,
     +    /, '0.240000 0.240000 scale')
c
c Write the title and the labels:
c
      ts   = 7.5
      xloc = hpxmin - 0.15*xrange
      yloc = hpymin + 0.50*yrange
      call pstext(xloc,yloc,24,ylab,ts,1,90.0,1)
      xloc = hpxmin + 0.50*xrange
      yloc = hpymin - 0.15*yrange
      call pstext(xloc,yloc,24,xlab,ts,1,0.0,1)
      yloc = hpymax + 0.01*yrange
      call pstext(hpxmin,yloc,40,title,8.0,3,0.0,0)
c
c Increase the max scaling a little for the labeling of the axes:
c
      xscmax = xscmax + 0.001*(xscmax-xscmin)
      yscmax = yscmax + 0.001*(yscmax-yscmin)
c
c Sort out the scaling for log axis:
c
      if(ilogx.eq.1) then
            xhsmn = 10.0**xscmin
            xhsmx = 10.0**xscmax
      else
            xhsmn = xscmin
            xhsmx = xscmax
      end if
      if(ilogy.eq.1) then
            yhsmn = 10.0**yscmin
            yhsmx = 10.0**yscmax
      else
            yhsmn = yscmin
            yhsmx = yscmax
      end if
                                    ilog = 0
      if(ilogx.eq.1)                ilog = 1
      if(ilogy.eq.1)                ilog = 2
      if(ilogx.eq.1.and.ilogy.eq.1) ilog = 3
c
c Scale and Draw the scatterplot axes:
c
      call scal(xhsmn,xhsmx,yhsmn,yhsmx,hpxmin,hpxmax,hpymin,
     +          hpymax,ilog,1)
c
c Locate all the points:
c
      np = 0
      do i=1,nd
         if((int(i/nth)*nth).eq.i) then
            if(vr1(i).ge.xhsmn.and.vr1(i).lt.xhsmx.and.
     +         vr2(i).ge.yhsmn.and.vr2(i).lt.yhsmx) then
                  if(ilogx.eq.1) then
                        xval = alog10(max(vr1(i),EPSLON))
                  else
                        xval = vr1(i)
                  endif
                  if(ilogy.eq.1) then
                        yval = alog10(max(vr2(i),EPSLON))
                  else
                        yval = vr2(i)
                  endif
                  np = np + 1
                  xx = resc(xscmin,xscmax,hpxmin,hpxmax,xval)
                  yy = resc(yscmin,yscmax,hpymin,hpymax,yval)
                  ix = int((resc(xmin,xmax,pxmin,pxmax,xx))/pscl)
                  iy = int((resc(ymin,ymax,pymin,pymax,yy))/pscl)
                  if(vr3(i).gt.-1.0e21) then
                        gray = 1.0 - (vr3(i)-gmin)/(gmax-gmin)
                        write(lpsout,102) ix,iy,bsize,gray
 102                    format('n ',i5,1x,i5,1x,f5.2,' 0 360 arc gs ',
     +                          f5.3,' setgray fill gr c')
                  else
                        write(lpsout,101) ix,iy
 101                    format('n ',i5,1x,i5,' bullet')
                  end if
            end if
         end if
      end do
c
c Calculate the rank correlation:
c
      xd = real(nd)
      call sortem(1,nd,vr1,1,vr2,c,d,e,f,g,h)
      do i=1,nd
            vr1(i) = real(i)
      end do
      call sortem(1,nd,vr2,1,vr1,c,d,e,f,g,h)
      rm = 0.0
      do i=1,nd
            rm = rm + vr1(i)
      end do
      rm = rm / xd
      rv = 0.0
      rr = 0.0
      do i=1,nd
            rv = rv + (vr1(i)-rm)*(vr1(i) -rm)
            rr = rr + (vr1(i)-rm)*(real(i)-rm)
      end do
      rv = rv / xd
      rr = rr / xd
      rr = rr / sqrt(max((rv*rv),0.0))
c
c Write the Statistics to the Output Device:
c
      x1  = 72.0
      x2  = 73.0
      yy  = 55.5
      y1  =  2.5
      y2  =  4.0
      it  =  1
      rt  =  0.0
      ts  =  7.0
c
      str = 'Number of data'
      call pstext(x1,yy,14,str,ts,it,rt,2)
      write(str(1:6),'(i6)') nd
      call pstext(x2,yy,6,str,ts,it,rt,0)
      yy = yy - y1
c
      str = 'Number plotted'
      call pstext(x1,yy,14,str,ts,it,rt,2)
      write(str(1:6),'(i6)') np
      call pstext(x2,yy,6,str,ts,it,rt,0)
      yy = yy - y1
c
      if(nt.gt.0) then
            str = 'Number trimmed'
            call pstext(x1,yy,14,str,ts,it,rt,2)
            write(str(1:6),'(i6)') nt
            call pstext(x2,yy,6,str,ts,it,rt,0)
      endif
      yy = yy - y2
c
      str = 'X Variable: mean'
      call pstext(x1,yy,16,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') vr1m
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y1
c
      str = 'std. dev.'
      call pstext(x1,yy,9,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') sqrt(max(vr1v,0.0))
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y2
c
      str = 'Y Variable: mean'
      call pstext(x1,yy,16,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') vr2m
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y1
c
      str = 'std. dev.'
      call pstext(x1,yy,9,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') sqrt(max(vr2v,0.0))
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y2
c
      str = 'correlation'
      call pstext(x1,yy,11,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') corr
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y1
c
      str = 'rank correlation'
      call pstext(x1,yy,16,str,ts,it,rt,2)
      write(str(1:12),'(f12.3)') rr
      call pstext(x2,yy,12,str,ts,it,rt,0)
      yy = yy - y2
c
      if(iwt.ge.1) then
            str = 'weights used'
            call pstext(x1,yy,12,str,ts,it,rt,2)
      end if
c
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
 9998 format(/' SCATPLT Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'Error in parameter file somewhere'
 99   stop 'Error in data file somewhere'
      end
c
c
c
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
      open(lun,file='scatplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for SCATPLT',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat               ',
     +       '-file with data')
      write(lun,12)
 12   format('4   3   0    0                    ',
     +       '-  columns for X, Y, wt, third var.')
      write(lun,13)
 13   format('-1.0   1.0e21                     ',
     +       '-  trimming limits')
      write(lun,14)
 14   format('scatplt.ps                        ',
     +       '-file for Postscript output')
      write(lun,15)
 15   format('0.0     20.0      0               ',
     +       '-X min and max, (0=arith, 1=log)')
      write(lun,16)
 16   format('0.0     20.0      0               ',
     +       '-Y min and max, (0=arith, 1=log)')
      write(lun,17)
 17   format('1                                 ',
     +       '-plot every nth data point')
      write(lun,18)
 18   format('1.0                               ',
     +       '-bullet size: 0.1(sml)-1(reg)-10(big)')
      write(lun,19)
 19   format('0.0      2.0                      ',
     +       '-limits for third variable gray scale')
      write(lun,20)
 20   format('Primary versus Secondary          ',
     +       '-title')

      close(lun)
      return
      end
