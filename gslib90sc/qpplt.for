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
c                        Q-Q, or P-P Plots
c                        *****************
c
c This program generates a PostScript file with a Q-Q or a P-P plot
c
c INPUT/OUTPUT Parameters:
c
c   datafl1     the input data for the first variable
c   ivr1,iwt1   columns for the variable and the weight (0 if none)
c   datafl2     the input data for the second variable
c   ivr2,iwt2   columns for the variable and the weight (0 if none)
c   tmin,tmax   trimming limts (acceptable: >= tmin and < tmax)
c   outfl       output file with histogram
c   qqorpp      0 = Q-Q plot, 1 = P-P plot
c   npts        number of points to label
c   min1,max1   plotting limts (will choose automatically if hmax<hmin)
c   min2,max2   plotting limts (will choose automatically if hmax<hmin)
c   title       title for output PostScript file
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter
c    file is described in the manual (see the example qpplt.par).
c
c 2. If data are <tmin , >=tmax , or the weight is <EPSLON the data
c    will not be considered.
c
c 3. Only the overlapping parts of the two distributions are plotted.
c    No extrapolation beyond the minimum and maximum data values is
c    considered.
c
c 4. Using large data sets for both variables will cause very large
c    printing times.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MV=500, EPSLON=1.0e-20, VERSION=3.000)

      character  datafl1*512,datafl2*512,outfl*512,title*40,str*512,
     +           xlab*24,ylab*24
      real       var(20)
      integer    qqorpp,test
      logical    testfl
c
c Dynamic allocation of the arrays.
c
      real, allocatable :: vr1(:),vr2(:),z1(:),p1(:),z2(:),p2(:)

      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax

      data lin/1/,lpsout/2/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +     pymin/0.0/,pymax/216.0/,xmin/-10.0/,xmax/83.3/,
     +     ymin/-10.0/,ymax/60.0/,hpxmin/1.0/,hpxmax/59.0/,
     +     hpymin/0.0/,hpymax/58.0/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' QPPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'qpplt.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'qpplt.par           ') then
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
      read(lin,'(a512)',err=97) datafl1
      call chknam(datafl1,512)
      write(*,*) ' first data file = ',datafl1(1:40)

      read(lin,*,err=97) ivr1,iwt1
      write(*,*) ' columns = ',ivr1,iwt1

      read(lin,'(a512)',err=97) datafl2
      call chknam(datafl2,512)
      write(*,*) ' second data file = ',datafl2(1:40)

      read(lin,*,err=97) ivr2,iwt2
      write(*,*) ' columns = ',ivr2,iwt2

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) qqorpp
      write(*,*) ' qqorpp flag = ',qqorpp

      read(lin,*,err=97) npoints
      write(*,*) ' number of points = ',npoints
      if(npoints.le.0) npoints = 1000000

      read(lin,*,err=97) xscmin,xscmax
      write(*,*) ' first axes scaling = ',xscmin,xscmax

      read(lin,*,err=97) yscmin,yscmax
      write(*,*) ' second axes scaling = ',yscmin,yscmax

      read(lin,*,err=97) ilog
      write(*,*) ' log scaling option = ',ilog
      if(ilog.gt.0)   ilog = 3
      if(qqorpp.eq.1) ilog = 0

      read(lin,'(a40)',err=97) title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      close(lin)
c
c Check to make sure the data file exists, then either write an error
c message and stop, or read in as much data as possible:
c
      inquire(file=datafl1,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the first data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      endif
c
c The first data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl1,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'(a24)',err=99) str(1:24)
            if(i.eq.ivr1) xlab = str(1:24)
      end do
      maxdat = 0
 20   read(lin,*,end=50,err=97) (var(j),j=1,nvari)
      if(var(ivr1).lt.tmin.or.var(ivr1).ge.tmax) go to 20
      maxdat = maxdat + 1
      go to 20
 50   continue
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
      allocate (z1(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (p1(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
c Now read the data for real:
c
      rewind(lin)
      read(lin,'(a)',err=99) str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'(a24)',err=99) str(1:24)
            if(i.eq.ivr1) xlab = str(1:24)
      end do
      if(ivr1.gt.nvari) then
            write(*,*) ' ivr is greater than the number in data file!'
            stop
      end if
c
c Read as much data as possible:
c
      v1min = 1.0e21
      v1max =-1.0e21
      n1  = 0
      nt1 = 0
 3    read(lin,*,end=4,err=99) (var(j),j=1,nvari)    
c
c Trim this data?
c
      if(var(ivr1).lt.tmin.or.var(ivr1).ge.tmax) then
            nt1 = nt1 + 1
            go to 3
      endif
      if(iwt1.ge.1) then
            if(var(iwt1).le.EPSLON) then
                  nt1 = nt1 + 1
                  go to 3
            endif
      endif
c
c Accept this data:
c
      n1 = n1 + 1
      z1(n1) = var(ivr1)
      v1min = min(z1(n1),v1min)
      v1max = max(z1(n1),v1max)
      if(iwt1.ge.1) then
            p1(n1) = var(iwt1)
      else
            p1(n1) = 1.0
      endif
c
c Go back for another data:
c
      go to 3
 4    close(lin)
c
c Read the second set of data:
c
      inquire(file=datafl2,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the second data file does not exist'
            write(*,*) '        check for the file and try again  '
            stop
      endif
c
c The second data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl2,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'(a24)',err=99) str(1:24)
            if(i.eq.ivr2) ylab = str(1:24)
      end do
      if(ivr2.gt.nvari) then
            write(*,*) ' ivr is greater than the number in data file!'
            stop
      end if
      maxdat = 0
 21   read(lin,*,end=51,err=97) (var(j),j=1,nvari)
      if(var(ivr2).lt.tmin.or.var(ivr2).ge.tmax) go to 21
      maxdat = maxdat + 1
      go to 21
 51   continue
c
c Allocate the needed memory.
c
      allocate (vr2(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (z2(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (p2(maxdat), stat = test)
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
            if(i.eq.ivr2) ylab = str(1:24)
      end do
      if(ivr2.gt.nvari) then
            write(*,*) ' ivr is greater than the number in data file!'
            stop
      end if
c
c Read as much data as possible:
c
      v2min = 1.0e21
      v2max =-1.0e21
      n2  = 0
      nt2 = 0
 7    read(lin,*,end=8,err=99) (var(j),j=1,nvari)
c
c Trim this data?
c
      if(var(ivr2).lt.tmin.or.var(ivr2).ge.tmax) then
            nt2 = nt2 + 1
            go to 7
      endif
      if(iwt2.ge.1) then
            if(var(iwt2).le.EPSLON) then
                  nt2 = nt2 + 1
                  go to 7
            endif
      endif
c
c Accept this data:
c
      n2 = n2 + 1
      z2(n2) = var(ivr2)
      v2min = min(z2(n2),v2min)
      v2max = max(z2(n2),v2max)
      if(iwt2.ge.1) then
            p2(n2) = var(iwt2)
      else
            p2(n2) = 1.0
      endif
c
c Go back for another data:
c
      go to 7
 8    close(lin)
c
c Report the number of data in each file:
c
      write(*,*)
      write(*,*)   ' First data file:  acceptable data = ',n1
      write(*,*)   '                   trimmed data    = ',nt1
      write(*,*)   ' Second data file: acceptable data = ',n2
      write(*,*)   '                   trimmed data    = ',nt2
      write(*,*)
c
c Create Cumulative Probabilities out of Variables:
c
c
c CDF for first data set:
c
      call sortem(1,n1,z1,1,p1,c,d,e,f,g,h)
      ccdf = 0
      do i=1,n1
            ccdf = ccdf + p1(i)
      end do
      if(ccdf.lt.EPSLON) stop 'Cumulative Probability too LOW'
      oldcp = 0.0
      cp    = 0.0
      do i=1,n1
            cp    = cp + p1(i)/ccdf
            p1(i) = 0.5*(cp+oldcp)
            oldcp = cp
      end do
c
c CDF for second data set:
c
      call sortem(1,n2,z2,1,p2,c,d,e,f,g,h)
      ccdf = 0
      do i=1,n2
            ccdf = ccdf + p2(i)
      end do
      if(ccdf.lt.EPSLON) stop 'Cumulative Probability too LOW'
      oldcp = 0.0
      cp    = 0.0
      do i=1,n2
            cp    = cp + p2(i)/ccdf
            p2(i) = 0.5*(cp+oldcp)
            oldcp = cp
      end do
c
c Set up either a Q-Q or a P-P plot:
c
      npoints = min(npoints,n1,n2)
      nd      = 0
      if(qqorpp.eq.0) then
c
c Q-Q plot:
c
            cpinc = 1.0 / real(npoints)
            cp    = -0.5*cpinc
            do i=1,npoints
                  cp = cp + cpinc
                  call locate(p2,n2,1,n2,cp,j2)
                  call locate(p1,n1,1,n1,cp,j1)
                  j2 = min((n2-1),max(1,j2))
                  j1 = min((n1-1),max(1,j1))
                  zz1 = z1(j1)+(z1(j1+1)-z1(j1))*(cp-p1(j1))
     +                / (p1(j1+1)-p1(j1))
                  zz2 = z2(j2)+(z2(j2+1)-z2(j2))*(cp-p2(j2))
     +                / (p2(j2+1)-p2(j2))
                  nd = nd + 1
                  vr1(nd) = zz1
                  vr2(nd) = zz2
            end do 
c
c P-P plot
c
      else
            zmin = max(z1(1), z2(1) )
            zmax = min(z1(n1),z2(n2))
            zinc = (zmax-zmin) / real(npoints+1)
            zz   = zmin - 0.5*zinc
            do i=1,npoints
                  zz = zz + zinc
                  call locate(z1,n1,1,n1,zz,j1)
                  call locate(z2,n2,1,n2,zz,j2)
                  j2 = min((n2-1),max(1,j2))
                  j1 = min((n1-1),max(1,j1))
                  cp1 = p1(j1)+(p1(j1+1)-p1(j1))*(zz-z1(j1))
     +                / (z1(j1+1)-z1(j1))
                  cp2 = p2(j2)+(p2(j2+1)-p2(j2))*(zz-z2(j2))
     +                / (z2(j2+1)-z2(j2))
                  nd = nd + 1
                  vr1(nd) = cp1
                  vr2(nd) = cp2
            end do 
      endif
c
c Limits for plot:
c
      if(xscmax.le.xscmin.or.yscmax.le.yscmin) then
            xscmin = min(vr1(1),vr2(1))
            xscmax = max(vr1(npoints),vr2(npoints))
            xrange = xscmax - xscmin
            xscmin = xscmin - 0.1*xrange
            xscmax = xscmax + 0.1*xrange
            yscmin = xscmin
            yscmax = xscmax
      endif
      if(qqorpp.eq.0) then
            if(ilog.gt.0) then
c
c Write an error message if minimum values too low:
c
                  if(vr1(1).lt.EPSLON.or.vr2(1).lt.EPSLON) then
                        write(*,*) 
                        write(*,*) 'You have asked for log scaling ',
     +                             'but data are out of bounds!'
                        write(*,*) vr1(1),vr2(1)
                        write(*,*) 
                  end if
                  xscmin = max(xscmin,vr1(1))
                  xscmax = max(xscmax,vr1(npoints))
                  xscmin = real(int(alog10(max(xscmin,0.00001))-0.9))
                  xscmax = real(int(alog10(max(xscmax,0.00001))+0.9))
                  yscmin = max(yscmin,vr2(1))
                  yscmax = max(yscmax,vr2(npoints))
                  yscmin = real(int(alog10(max(yscmin,0.00001))-0.9))
                  yscmax = real(int(alog10(max(yscmax,0.00001))+0.9))
                  xscmin = min(xscmin,yscmin)
                  yscmin = min(xscmin,yscmin)
                  xscmax = min(xscmax,yscmax)
                  yscmax = min(xscmax,yscmax)
            endif
      else
            xscmin = 0.0
            xscmax = 1.05
            yscmin = xscmin
            yscmax = xscmax
      end if
c
c Some initialization:
c
      xrange = hpxmax - hpxmin
      yrange = hpymax - hpymin
c
c Open the output file and add a header:
c
      open(lpsout,file=outfl,status='UNKNOWN')
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
c Sort out the scaling for log axis:
c
      if(ilog.gt.0) then
            xhsmn = 10.0**xscmin
            xhsmx = 10.0**xscmax
            yhsmn = 10.0**yscmin
            yhsmx = 10.0**yscmax
      else
            xhsmn = xscmin
            xhsmx = xscmax
            yhsmn = yscmin
            yhsmx = yscmax
      end if
c
c Scale and Draw the scatterplot axes:
c
      call scal(xhsmn,xhsmx,yhsmn,yhsmx,hpxmin,hpxmax,hpymin,hpymax,
     +          ilog,1)
c
c Locate all the points:
c
      do i=1,nd
            if(vr1(i).ge.xhsmn.and.vr1(i).lt.xhsmx.and.
     +         vr2(i).ge.yhsmn.and.vr2(i).lt.yhsmx) then
                  if(ilog.gt.0) then
                        xval = alog10(max(vr1(i),EPSLON))
                        yval = alog10(max(vr2(i),EPSLON))
                  else
                        xval = vr1(i)
                        yval = vr2(i)
                  endif
                  xx = resc(xscmin,xscmax,hpxmin,hpxmax,xval)
                  yy = resc(yscmin,yscmax,hpymin,hpymax,yval)
                  ix = int((resc(xmin,xmax,pxmin,pxmax,xx))/pscl)
                  iy = int((resc(ymin,ymax,pymin,pymax,yy))/pscl)
                  write(lpsout,101) ix,iy
 101              format('n ',i5,1x,i5,' bullet')
            end if
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
 9998 format(/' QPPLT Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'Error in parameter file somewhere'
 99   stop 'Error in data file somewhere'
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
      open(lun,file='qpplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for QPPLT',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat              ',
     +       '-file with first set of data (X axis)')
      write(lun,12)
 12   format('3   0                            ',
     +       '-  columns for variable and weight')
      write(lun,13)
 13   format('../data/data.dat                 ',
     +       '-file with second set of data (Y axis)')
      write(lun,14)
 14   format('3   0                            ',
     +       '-  columns for variable and weight')
      write(lun,15)
 15   format('-1.0       1.0e21                ',
     +       '-  trimming limits')
      write(lun,16)
 16   format('qpplt.ps                         ',
     +       '-file for PostScript output')
      write(lun,17)
 17   format('0                                ',
     +       '-0=Q-Q plot, 1=P-P plot')
      write(lun,18)
 18   format('0                                ',
     +       '-number of points to plot (<0 for all)')
      write(lun,19)
 19   format('0.0     20.0                     ',
     +       '-X minimum and maximum')
      write(lun,20)
 20   format('0.0     20.0                     ',
     +       '-Y minimum and maximum')
      write(lun,21)
 21   format('0                                ',
     +       '-0=arithmetic, 1=log scaling')
      write(lun,22)
 22   format('Small Data Set Versus Clustered Data  ',
     +       '-Title')

      close(lun)
      return
      end
