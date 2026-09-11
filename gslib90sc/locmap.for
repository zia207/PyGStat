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
c                    Location Map of 2-D Data
c                    ************************
c
c This program generates a PostScript file with a 2-D gray/color scale
c posting of data locations
c
c INPUT/OUTPUT Parameters:\
c
c   datafl         the input data
c   ix,iy,ivr      column numbers for X, Y, and value
c   tmin,tmax      values outside this range are considered missing
c   outfl          output file with PostScript plot
c   xmn,xmx        for min and max in X direction
c   ymn,ymx        for min and max in Y direction
c   icros          0=not cross validation, 1=yes
c   ilog           0=arithmetic, 1=logarithmic
c   icolor         0=gray scale, 1=color scale
c   ilabel         0=no labels,  1=add labels
c   gmin,gmax      scaling for continous min, max
c   labsiz         scaling for label size (1.0 is default)
c   title          title for output PostScript file
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter 
c    file is described in the documentation (see the example locmap.par)
c
c
c
c The following Parameters control static dimensioning:
c
c   MAXH      maximum resolution of scalebar
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MAXH=500,BIGNUM=1.0e20,EPSLON=1.0e-20,VERSION=2.906)

      character datafl*512,outfl*512,title*40,str*512,iline*2,
     +          jline(MAXH)*2,hexa*2,rline(MAXH)*2,gline(MAXH)*2,
     +          bline(MAXH)*2
      real      xx(5),yy(5),var(50)
      logical   testfl,showlin
      integer   test
c
c Common variables for Postscript Plotting:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      common /color/  cmin,cmax,cint(4),cscl
c
c Dynamic allocation of the arrays.
c
      real, allocatable :: x(:),y(:),vr(:)
c
c Hardwire many of the plot parameters:
c
      data       lin/1/,lpsout/1/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +           pymin/0.0/,pymax/216.0/,xmin/0.0/,xmax/80.0/,
     +           ymin/0.0/,ymax/60.0/,siz/25.0/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' LOCMAP Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'locmap.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'locmap.par          ') then
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

      read(lin,*,err=97) ixl,iyl,ivrl
      write(*,*) ' column numbers = ',ixl,iyl,ivrl

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) xmn,xmx
      write(*,*) ' xmn, xmx = ',xmn,xmx

      read(lin,*,err=97) ymn,ymx
      write(*,*) ' ymn, ymx = ',ymn,ymx

      read(lin,*,err=97) icros
      write(*,*) ' icros = ',icros

      read(lin,*,err=97) ilog
      write(*,*) ' ilog = ',ilog

      read(lin,*,err=97) icolor
      write(*,*) ' icolor = ',icolor

      read(lin,*,err=97) ilabel
      write(*,*) ' ilabel = ',ilabel

      showlin = .true.
      if(ilabel.eq.-1.0) then
            showlin = .false.
            ilabel  = 0.0
      end if

      read(lin,*,err=97) gmin,gmax,ginc
      write(*,*) ' gmin, gmax, ginc = ',gmin,gmax,ginc

      read(lin,*,err=97) sizfac
      write(*,*) ' label size = ',sizfac
      siz = siz * sizfac
      if(icros.eq.1) siz = siz * 1.5

      read(lin,'(a40)',err=97) title
      call chktitle(title,40)
      write(*,*) ' title = ',title
      close(lin)
c
c Read in the data (if the file exists): 
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      endif
c
c The first data file exists so open the file and read in the header
c information. Find MAXDAT and allocate the storage that will be used 
c to summarize the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,'(a)',err=99)str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      maxdat = 0
 213  read(lin,*,end=400,err=97) (var(j),j=1,nvari)
      if(var(ivrl).lt.tmin.or.var(ivrl).ge.tmax) go to 213
      maxdat = maxdat + 1
      go to 213
 400  continue
c      
c
c
      allocate (x(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c            
c  
      allocate (y(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c            
      allocate (vr(maxdat), stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c            
      rewind(lin)
      nd = 0
      read(lin,'(a)',err=99)str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
 212  read(lin,*,end=3) (var(j),j=1,nvari)
      vrt = var(ivrl)
      if(vrt.ge.tmin.and.vrt.lt.tmax) then
            nd     = nd + 1
            x(nd)  = var(ixl)
            y(nd)  = var(iyl)
            vr(nd) = vrt
      end if
      go to 212
 3    close(lin)
c
c If necessary find minimum and maximum for plotting:
c
      if(gmin.ge.gmax) then
            gmin = BIGNUM
            gmax =-BIGNUM
            do i=1,nd
                  if(vr(i).lt.gmin) gmin = vr(i)
                  if(vr(i).gt.gmax) gmax = vr(i)
            end do
            test = 0.02*(gmax-gmin)
            if(gmin.le.test) gmin = 0.0
            gmax = int(gmax+0.99+0.01*(gmax-gmin))
      endif
c
c Add a header:
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
c Reset to a log_10 scaling?
c
      cinc = ginc
      if(ilog.eq.1) then
            gmin = alog10(max(gmin,EPSLON))
            gmax = alog10(max(gmax,EPSLON))
            cinc = 1.0
      end if
      cmin = gmin
      cmax = gmax
c
c Set up to draw a color image?
c
      if(icolor.eq.1) then
            crange  = cmax - cmin
            c0      = 0.00001
            h       = (crange + c0*crange)/4.0
            cint(1) = cmin + c0 * crange
            cint(2) = cint(1) + h
            cint(3) = cint(2) + h
            cint(4) = cint(3) + h
            cscl    = 255*(cint(1)-cmin)/h
      end if
c
c Decide on the best scaling parameters (do not distort):
c
      xtrans =  30.0
      xgmax  = 840.0
      ytrans =  30.0
      ygmax  = 840.0
      if((xmx-xmn).gt.(ymx-ymn)) then
            ytrans = 30.0 + 0.5*(810.0-(ymx-ymn)/(xmx-xmn)*810.0)
            ygmax  = ytrans + (ymx-ymn)/(xmx-xmn)*810.0
      else
            xtrans = 30.0 + 0.5*(810.0-(xmx-xmn)/(ymx-ymn)*810.0)
            xgmax  = xtrans + (xmx-xmn)/(ymx-ymn)*810.0
      endif
c
c Now set up and draw the gray scale location map:
c
      write(lpsout,100)
 100  format('%',/,'% START OF LOCATION MAP: ',/,'%')
      tsiz = 40 * sizfac
      write(lpsout,101) tsiz
 101  format('1.0 setlinewidth ',
     +       '/Helvetica findfont ',f5.1,' scalefont setfont')
      do i=1,nd
c
c Only plot those points within the area of interest:
c
            if(x(i).lt.xmn.or.x(i).gt.xmx) go to 7
            if(y(i).lt.ymn.or.y(i).gt.ymx) go to 7
            ix     = int(resc(xmn,xmx,xtrans,xgmax,x(i))+0.5)
            iy     = int(resc(ymn,ymx,ytrans,ygmax,y(i))+0.5)
            vallab = vr(i)

            if(ilog.eq.1) then
                  value = alog10(max(vr(i),EPSLON))
            else
                  value = vr(i)
            endif
            if(icros.eq.1)    value = abs(value)
            if(value.lt.gmin) value = gmin
            if(value.gt.gmax) value = gmax

            if(icros.eq.0) then
                        call putsym(lpsout,ix,iy,siz,1)
            else
                  if(vr(i).le.0.0) then
                        call putsym(lpsout,ix,iy,siz,2)
                  else
                        call putsym(lpsout,ix,iy,siz,3)
                  end if
            end if

            if(icolor.eq.0) then
                  gray = 1.0 - (value-gmin)/(gmax-gmin)
                  if(showlin) then
                        write(lpsout,203) gray
                  else
                        write(lpsout,204) gray
                  end if
 203              format(f4.2,' setgray fill grestore s')
 204              format(f4.2,' setgray fill grestore ')
            else
                  call red  (value,iline,rfrac)
                  call green(value,iline,gfrac)
                  call blue (value,iline,bfrac)
                  if(showlin) then
                        write(lpsout,113) rfrac,gfrac,bfrac
                  else
                        write(lpsout,114) rfrac,gfrac,bfrac
                  end if
 113              format(f6.4,2f7.4,' setrgbcolor fill grestore s')
 114              format(f6.4,2f7.4,' setrgbcolor fill grestore ')
            endif
c
c Add Label if requested:
c
            if(ilabel.eq.1) then
                  iy = iy + 20
                  call numtext(vallab,str(1:12))
                  lostr = 12
                  lost  = lostr
                  do ii=1,lostr
                        if(str(1:1).eq.' ') then
                              lost = lost - 1
                              do j=1,lost
                                    k = j + 1
                                    str(j:j) = str(k:k)
                              end do
                        else
                              go to 4
                        endif
                  end do
 4                continue
                  if(lost.gt.0) then
                        write(lpsout,116) ix,iy,str(1:lost)
 116                    format(i4,1x,i4,1x,' m (',a,') ctext')
                  end if
            end if
c
c Keep looping over all of the data points:
c
 7          continue
      end do
c
c Now draw the border:
c
      write(lpsout,104)
 104  format('%',/,'% BORDER: ',/,'%')
      thic = 1.0
      npts = 5
      xx(1) = xtrans / 15.0
      yy(1) = ytrans / 15.0
      xx(2) = xtrans / 15.0
      yy(2) = ygmax  / 15.0
      xx(3) = xgmax  / 15.0
      yy(3) = ygmax  / 15.0
      xx(4) = xgmax  / 15.0
      yy(4) = ytrans / 15.0
      xx(5) = xtrans / 15.0
      yy(5) = ytrans / 15.0
      call psline(npts,xx,yy,thic,0)
c
c Now draw the title:
c
      write(lpsout,105)
 105  format('%',/,'% TITLE: ',/,'%')
      xx(1) = (450.0        ) / 15.0
      yy(1) = (ygmax + 30.0 ) / 15.0
      call pstext(xx(1),yy(1),40,title,10.0,3,0.0,1)
c
c Scale and Draw the scatterplot axes:
c
      hpxmin = xtrans / 15.0
      hpxmax = xgmax  / 15.0
      hpymin = ytrans / 15.0
      hpymax = ygmax  / 15.0
      call scal(xmn,xmx,ymn,ymx,hpxmin,hpxmax,hpymin,hpymax,0,0)
c
c DRAW A LEGEND BAR:
c
      write(lpsout,123)
 123  format('%',/,'% START OF LEGEND: ',/,'%')
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
                  jline(ih) = hexa(level)
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
      nh = num
      xtrans = 930.0
      hscale =  90.0
      ytrans = 111.0
      vscale = 648.0
c
c
c
      if(icolor.eq.0) then
            write(lpsout,201)
 201        format('/gsmap{<')
            write(lpsout,'(200a2)') (jline(i),i=1,nh)
            write(lpsout,102) xtrans,ytrans,hscale,vscale
 102        format('>} def gsave ',2f9.3,' translate ',2f8.2,' scale')
            write(lpsout,103) 1,nh,1,nh
 103        format(2i4,' 8 [',i3,' 0 0 ',i3,' 0 0] {gsmap}',
     +             ' image grestore')
      else
            write(lpsout,205)
            write(lpsout,'(200a2)') (rline(i),i=1,nh)
            write(lpsout,206)
            write(lpsout,207)
            write(lpsout,'(200a2)') (gline(i),i=1,nh)
            write(lpsout,206)
            write(lpsout,208)
            write(lpsout,'(200a2)') (bline(i),i=1,nh)
            write(lpsout,206)
            write(lpsout,209) xtrans,ytrans,hscale,vscale
            write(lpsout,210) 1,nh,1,nh
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
      write(lpsout,115)
 115  format('%',/,'% BORDER AROUND LEGEND: ',/,'%',/,
     +       '0 0 0 setrgbcolor')
      thic = 0.3
      npts = 5
      xx(1) = (xtrans          ) / 15.0
      yy(1) = (ytrans          ) / 15.0
      xx(2) = (xtrans          ) / 15.0
      yy(2) = (ytrans + vscale ) / 15.0
      xx(3) = (xtrans + hscale ) / 15.0
      yy(3) = (ytrans + vscale ) / 15.0
      xx(4) = (xtrans + hscale ) / 15.0
      yy(4) = (ytrans          ) / 15.0
      xx(5) = (xtrans          ) / 15.0
      yy(5) = (ytrans          ) / 15.0
      call psline(npts,xx,yy,thic,0)
c
c Label the legend:
c
      num   = int((cmax-cmin)/cinc+0.5)
      yinc  = vscale / real(num)
      cval  = cmin - cinc
      xval  = 1035.0
      yval  = ytrans + 2.0 - yinc
      xx(1) = 1020.0 / 15.0
      xx(2) = 1030.0 / 15.0
      yy(1) = (ytrans - yinc) /15.0
      yy(2) = yy(1)
      npts  = 2
      tsiz  = 4.0 + 3.0 * (vscale/672.0)
      do i=1,num+1
            cval = cval + cinc
            if(ilog.eq.1) then
                  value = 10**cval
            else
                  value = cval
            endif
            call numtext(value,title(1:12))
            yval = yval + yinc
            call pstext(xval/15.0,yval/15.0,12,title,tsiz,1,0.0,0)
            yy(1) = yy(1) + yinc / 15.0
            yy(2) = yy(1)
            call psline(npts,xx,yy,thic,0)
      end do
c
c Closing on PostScript File:
c
      write(lpsout,999)
 999  format('%END OF POSTSCRIPT FILE',/,'4.166667 4.166667 scale',/,/,
     +       '%%EOF',/,'showpage')
c
c Finished:
c
      close(lpsout)
      write(*,9998) VERSION
 9998 format(/' LOCMAP Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in parameters somewhere'
 99   stop 'ERROR in data somewhere'
      end



      subroutine putsym(lout,ix,iy,siz,itype)
c----------------------------------------------------------------------
c 
c Add Appropriate Symbol (circle, minus or plus sign) at Data Location
c ********************************************************************
c
c This subroutine will add a symbol "itype" with "siz" centered at  the 
c position "ix","iy" in the output file "lout". 
c
c itype = 1:  a filled circle;
c itype = 2:  a minus sign;
c itype = 3:  a plus sign.
c
c 
c-----------------------------------------------------------------------
      integer lout,ix,iy,itype,ih,iv,ib,is
      real    siz
c
c Circle
c
      if(itype.eq.1) then
            write(lout,101) ix,iy,int(siz)
 101        format('n ',i4,1x,i4,1x,i2,1x,'0 360 arc c gsave')
c
c Minus sign
c
      else if(itype.eq.2) then
            ih = int(siz/2.0)
            iv = int(siz/4.0)
            write(lout,201) (ix-ih),(iy-iv),(ix-ih),(iy+iv),
     +                      (ix+ih),(iy+iv),(ix+ih),(iy-iv)
 201        format('n',2i5,' m',2i5,' l',2i5,' l',2i5,' l c gsave')
c
c Plus sign
c
      else if(itype.eq.3) then
            ib = int(siz/2.0)
            is = int(siz/6.0)
            write(lout,301) (ix-is),(iy-is),(ix-ib),(iy-is),
     +                      (ix-ib),(iy+is),(ix-is),(iy+is) 
            write(lout,302) (ix-is),(iy+ib),(ix+is),(iy+ib),
     +                      (ix+is),(iy+is),(ix+ib),(iy+is)
            write(lout,303) (ix+ib),(iy-is),(ix+is),(iy-is),
     +                      (ix+is),(iy-ib),(ix-is),(iy-ib)
 301        format('n',2i5,' m',2i5,' l',2i5,' l',2i5,' l')
 302        format(    2i5,' l',2i5,' l',2i5,' l',2i5,' l')
 303        format(    2i5,' l',2i5,' l',2i5,' l',2i5,' l c gsave')
      end if
c
c Return
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
      open(lun,file='locmap.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for LOCMAP',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat             ',
     +       '-file with data')
      write(lun,12)
 12   format('1   2   3                       ',
     +       '-   columns for X, Y, variable')
      write(lun,13)
 13   format('-1.0    1.0e21                  ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('locmap.ps                       ',
     +       '-file for PostScript output')
      write(lun,15)
 15   format('0.0     50.                     ',
     +       '-xmn,xmx')
      write(lun,16)
 16   format('0.0     50.                     ',
     +       '-ymn,ymx')
      write(lun,17)
 17   format('0                               ',
     +       '-0=data values, 1=cross validation')
      write(lun,18)
 18   format('1                               ',
     +       '-0=arithmetic,  1=log scaling')
      write(lun,19)
 19   format('1                               ',
     +       '-0=gray scale,  1=color scale')
      write(lun,20)
 20   format('0                               ',
     +       '-0=no labels,   1=label each location')
      write(lun,21)
 21   format('0.01   10.0    1.               ',
     +       '-gray/color scale: min, max, increm')
      write(lun,22)
 22   format('0.5                             ',
     +       '-label size: 0.1(sml)-1(reg)-10(big)')
      write(lun,23)
 23   format('Locations of Clustered Data     ',
     +       '-Title')

      close(lun)
      return
      end
