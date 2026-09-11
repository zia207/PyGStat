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
c                     Gray/Color Scale of a 2-D slice
c                     *******************************
c
c This program generates a PostScript file with a 2-D gray/color scale
c map of a continuous or categorical variable grid.
c
c INPUT/OUTPUT Parameters:
c
c   datafl      the input data
c   ivr         column number in the input file
c   tmin,tmax   values outside this range are considered missing
c   outfl       output file with PostScript map
c   igrid       grid number
c   nx,xmn,xsiz grid definition in X direction
c   ny,ymn,ysiz grid definition in Y direction
c   nz,zmn,zsiz grid definition in Z direction
c   iview       view (1=XY, 2=XZ, 3=YZ)
c   islice      slice number
c   title       title for output PostScript file
c   xlab,ylab   label for X and Y axes
c   ilog        0=arithmetic, 1=logarithmic
c   icolor      0=gray scale, 1=color scale
c   icat        0=continuous, 1=categorical
c   cmin,cmax,cinc  scaling for continous min, max, label increment
c   ncat            number of categories
c   cat(), ccode()  category number and code number
c
c
c
c PROGRAM NOTES:
c
c 1. The program is executed with no command line arguments.  The user
c    will be prompted for the name of a parameter file.  The parameter 
c    file is described in the documentation (see the example cscale.par)
c
c
c
c The following Parameters control static dimensioning:
c
c   MAXH      maximum nodes in Horizontal or X direction
c   MAXV      maximum nodes in Vertical or Y direction
c
c
c
c Based on program   ``gscale.f''  and  ``color.f''  provided by Hua Zhu
c Modified: Clayton V. Deutsch                           Date: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MAXH   =1000, MAXV=1000,BIGNUM =99999., EPSLON=1.0e-20,
     +          MAXCAT =  24, VERSION=3.000)

      integer   icode(MAXCAT),ccode(MAXCAT),redint(MAXCAT),
     +          grnint(MAXCAT),bluint(MAXCAT)
      real      val(MAXH,MAXV),xx(5),yy(5),var(100)
      character datafl*512,outfl*512,title*40,xlabel*40,ylabel*40,
     +          str*512,iline(MAXH)*2,rline(MAXH)*2,gline(MAXH)*2,
     +          bline(MAXH)*2,cname(MAXCAT)*12,hexa*2
      logical   testfl,skipleg
c
c Common variables for Postscript Plotting:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      common /color/  cmin,cmax,cint(4),cscl
c
c Hardcoded categorical colors:
c
      data       redint/255,255,255,127,  0,  0,  0,255,255,0,127,170,
     +                  255,  0,200,9*255/,
     +           grnint/  0,127,255,255,255,255,  0,  0,255,0,  0, 85,
     +                   85,255,200,9*  0/,
     +           bluint/  0,  0,  0,  0,  0,255,255,255,255,0,255,  0,
     +                  170,127,200,9*255/
c
c Hardwire many of the plot parameters:
c
      data  lin/1/,lpsout/1/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +      pymin/0.0/,pymax/216.0/,xmin/0.0/,xmax/1200.0/,
     +      ymin/0.0/,ymax/900.0/,xmn/0.0/,ymn/0.0/,c0/0.00001/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' PIXELPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'pixelplt.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'pixelplt.par        ') then
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

      read(lin,*,err=97) ivrc
      write(*,*) ' column number = ',ivrc

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) igrid
      write(*,*) ' realization number = ',igrid

      read(lin,*,err=97) nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz = ',nx,xmn,xsiz

      read(lin,*,err=97) ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz = ',ny,ymn,ysiz

      read(lin,*,err=97) nz,zmn,zsiz
      write(*,*) ' nz, zmn, zsiz = ',nz,zmn,zsiz

      skipleg = .false.
      read(lin,*,err=97) iview
      if(iview.lt.0) then
            skipleg = .true.
            iview = abs(iview)
      end if
      write(*,*) ' slice type = ',iview

      read(lin,*,err=97) islice
      write(*,*) ' slice number = ',islice
c
c If the slice number greater than the grid number in the direction
c orthogonal ot the slice orientation, then give a error message
c 
      if( ((iview.eq.1).and.(islice.gt.nz)).or.
     +    ((iview.eq.2).and.(islice.gt.ny)).or.
     +    ((iview.eq.3).and.(islice.gt.nx)))    then
           write(*,*) ' '
           write(*,*) 'The slice number should not be larger than the'
           write(*,*) 'grid number orthogonal to the slice orientation!'
           write(*,*) 'Error in the parameter file!'
           write(*,*) '                            '
           stop
      endif

      read(lin,'(a40)',err=97) title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      read(lin,'(a40)',err=97) xlabel
      call chktitle(xlabel,40)
      write(*,*) ' X axes label = ',xlabel

      read(lin,'(a40)',err=97) ylabel
      call chktitle(ylabel,40)
      write(*,*) ' Y axes label = ',ylabel

      read(lin,*,err=97) ilog
      write(*,*) ' log scaling flag = ',ilog

      read(lin,*,err=97) icolor
      write(*,*) ' color scaling flag = ',icolor

      read(lin,*,err=97) icat
      write(*,*) ' categorical variable flag = ',icat

      read(lin,*,err=97) cmin,cmax,cinc
      write(*,*) ' cmin, cmax, cinc = ',cmin,cmax,cinc
c
c Make sure that these numbers are reasonable:
c
      if(cmin.ge.cmax) then
            write(*,*)
            write(*,*) 'NO automatic scaling of cmin and cmax '
            write(*,*) '   set cmin less than cmax and try again'
            stop
      end if
      num = int((cmax-cmin)/cinc+0.5)
      if(icat.eq.0.and.ilog.eq.0.and.(num.le.0.or.num.gt.20.0)) then
            write(*,*)
            write(*,*) 'BAD choice of cinc for cmin and cmax '
            write(*,*) '    there would be ',num,' labels '
            write(*,*) '    reset cinc and try again'
            cinc = (cmax-cmin)/10.0
            num = int((cmax-cmin)/cinc+0.5)
      end if
c
c If we are dealing with a categorical image:
c
      if(icat.eq.1) then
            read(lin,*,err=97) ncat
            write(*,*) ' number of categories = ',ncat
            if(ncat.gt.MAXCAT) stop 'too many categories!'
            do i=1,ncat
                  read(lin,*,err=97) icode(i),ccode(i),cname(i)
                  write(*,*) ' category ',i,' cat and code = ',
     +                       icode(i),ccode(i),' name: ',cname(i)
            end do
      end if
      close(lin)
c
c Check dimensioning:
c
      if(iview.eq.1) then
            nh   = nx
            hsiz = xsiz
            nv   = ny
            vsiz = ysiz
            hmn = xmn - 0.5 * xsiz
            vmn = ymn - 0.5 * ysiz
            hmx = xmn + (real(nx)-0.5) * xsiz
            vmx = ymn + (real(ny)-0.5) * ysiz
            if(nx.gt.MAXH) stop 'MAXH too small'
            if(ny.gt.MAXV) stop 'MAXV too small'
      else if(iview.eq.2) then
            nh   = nx
            hsiz = xsiz
            nv   = nz
            vsiz = zsiz
            hmn = xmn - 0.5 * xsiz
            vmn = zmn - 0.5 * zsiz
            hmx = xmn + (real(nx)-0.5) * xsiz
            vmx = zmn + (real(nz)-0.5) * zsiz
            if(nx.gt.MAXH) stop 'MAXH too small'
            if(nz.gt.MAXV) stop 'MAXV too small'
      else if(iview.eq.3) then
            nh   = ny
            hsiz = ysiz
            nv   = nz
            vsiz = zsiz
            hmn = ymn - 0.5 * ysiz
            vmn = zmn - 0.5 * zsiz
            hmx = ymn + (real(ny)-0.5) * ysiz
            vmx = zmn + (real(nz)-0.5) * zsiz
            if(ny.gt.MAXH) stop 'MAXH too small'
            if(nz.gt.MAXV) stop 'MAXV too small'
      else
            stop 'Invalid view code'
      end if
c
c Read in the data (if the file exists): 
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      endif
      open(lin,file=datafl,status='OLD')
      read(lin,'(a)',err=99) str
      read(lin,*,err=99)     nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      if(ivrc.gt.nvari) then
            write(*,*) ' ivr is greater than the number in data file!'
            stop
      end if
c
c Get the correct slice:
c
      if(igrid.gt.1) then
            do ig=1,igrid-1
                  do iz=1,nz
                        do iy=1,ny
                              do ix=1,nx
                                    read(lin,*)
                              end do
                        end do
                  end do
            end do
      end if
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        read(lin,*,err=99) (var(i),i=1,nvari)
                        value = var(ivrc)
                        if(iview.eq.1) then
                              if(iz.eq.islice) val(ix,iy) = value
                        else if(iview.eq.2) then
                              if(iy.eq.islice) val(ix,iz) = value
                        else if(iview.eq.3) then
                              if(ix.eq.islice) val(iy,iz) = value
                        end if
                  end do
            end do
      end do
      close(lin)
c
c Decide on the best scaling parameters (do not distort):
c
      htrans =  30.0
c old hgmax  = 840.0
      hgmax  = 810.0
      vtrans =  30.0
c old vgmax  = 840.0
      vgmax  = 810.0
      if((hmx-hmn).gt.(vmx-vmn)) then
            vtrans = 30.0 + 0.5*(810.0-(vmx-vmn)/(hmx-hmn)*810.0)
c - old     vgmax  = 30.0 +            (vmx-vmn)/(hmx-hmn)*810.0
            vgmax  =                   (vmx-vmn)/(hmx-hmn)*810.0
      else
            htrans = 30.0 + 0.5*(810.0-(hmx-hmn)/(vmx-vmn)*810.0)
c - old     hgmax  = 30.0 +            (hmx-hmn)/(vmx-vmn)*810.0
            hgmax  =                   (hmx-hmn)/(vmx-vmn)*810.0
      endif
c
c Open output file:
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
      if(ilog.eq.1) then
            cmin = alog10(max(cmin,0.0000001))
            cmax = alog10(max(cmax,0.0000001))
            cinc = 1.0
      end if
c
c Set up to draw a color image?
c
      if(icolor.eq.1.and.icat.eq.0) then
            crange  = cmax - cmin
            h       = (crange + c0*crange)/4.0
            cint(1) = cmin + c0 * crange
            cint(2) = cint(1) + h
            cint(3) = cint(2) + h
            cint(4) = cint(3) + h
            cscl    = 255*(cint(1)-cmin)/h
      end if
      write(lpsout,100)
 100  format('%',/,'% START OF IMAGE: ',/,'%')
c
c MAIN Loop over the rows:
c
      vgmax1 = vgmax / real(nv)
      vgsize = 1.001*vgmax1
      vloc   = vtrans - vgmax1
      do iv=1,nv
            vloc = vloc + vgmax1
c
c Hexadecimal representation of this row:
c
            do ih=1,nh
                  if(val(ih,iv).lt.tmin.or.val(ih,iv).ge.tmax) then
                        iline(ih) = 'FF'
                        rline(ih) = 'FF'
                        gline(ih) = 'FF'
                        bline(ih) = 'FF'
                  else
                        if(ilog.eq.1.and.icat.eq.0) then
                              value = alog10(max(val(ih,iv),0.0000001))
                        else
                              value =        val(ih,iv)
                        endif
                        if(icolor.eq.0.and.icat.eq.0) then
                              level = 256 - 256*(value-cmin)/(cmax-cmin)
                              level = max(min(level,255),0)
                              iline(ih) = hexa(level)
                        else
                              if(icat.eq.0) then
                                    value=min(cmax,max(cmin,value))
                                    call red  (value,rline(ih),rfrac)
                                    call green(value,gline(ih),gfrac)
                                    call blue (value,bline(ih),bfrac)
                              else
                                    ii = int(value+0.5)
                                    rline(ih) = hexa(redint(9))
                                    gline(ih) = hexa(grnint(9))
                                    bline(ih) = hexa(bluint(9))
                                    if(ii.gt.icode(ncat))ii=icode(ncat)
                                    do i=1,ncat
                                       if(ii.eq.icode(i)) then
                                           jj = ccode(i)
                                           rline(ih) = hexa(redint(jj))
                                           gline(ih) = hexa(grnint(jj))
                                           bline(ih) = hexa(bluint(jj))
                                           go to 91
                                       end if
                                    end do
                                    write(*,*) 'Undefined category: ',ii
 91                                 continue
                              end if
                        end if
                  end if
            end do
c
c Write this row to the PostScript file:
c
            if(icolor.eq.0.and.icat.eq.0) then
                  write(lpsout,101)
 101              format('/gsmap{<')
                  write(lpsout,'(200a2)') (iline(i),i=1,nh)
                  write(lpsout,102) htrans,vloc,hgmax,vgsize
 102              format('>} def gsave ',2f9.3,' translate ',2f8.2,
     +                   ' scale')
                  write(lpsout,103) nh,1,nh,1
 103              format(2i4,' 8 [',i5,' 0 0 ',i5,' 0 0] {gsmap}',
     +                   ' image grestore')
            else
                  write(lpsout,105)
 105              format('/rcl{<')
                  write(lpsout,'(200a2)') (rline(i),i=1,nh)
                  write(lpsout,106)
 106              format('>} def')
                  write(lpsout,107)
 107              format('/gcl{<')
                  write(lpsout,'(200a2)') (gline(i),i=1,nh)
                  write(lpsout,106)
                  write(lpsout,108)
 108              format('/bcl{<')
                  write(lpsout,'(200a2)') (bline(i),i=1,nh)
                  write(lpsout,106)
                  write(lpsout,109) htrans,vloc,hgmax,vgsize
 109              format('gsave ',2f9.3,' translate ',2f8.2,'  scale')
                  write(lpsout,110) nh,1,nh,1
 110              format(2i4,' 8 [',i5,' 0 0 ',i5,' 0 0] {rcl} {gcl} ',
     +                   '{bcl} true 3 colorimage grestore')
            endif
c
c END Loop over rows:
c
      end do
c
c Now draw the border:
c
      write(lpsout,111)
 111  format('%',/,'% BORDER: ',/,'%',/,'0 0 0 setrgbcolor')
      thic = 0.24
      npts = 5
      xx(1) = htrans
      yy(1) = vtrans
      xx(2) = xx(1)
      yy(2) = vtrans + vgmax
      xx(3) = htrans + hgmax
      yy(3) = yy(2)
      xx(4) = xx(3)
      yy(4) = yy(1)
      xx(5) = xx(1)
      yy(5) = yy(1)
      call psline(npts,xx,yy,thic,0)
c
c Now draw the title and labels:
c
      write(lpsout,112)
 112  format('%',/,'% TITLE: ',/,'%')
      xx(1) = 450.0
      yy(1) = vtrans + vgmax + 12.5
      call pstext(xx(1),yy(1),40,title,10.0,3,0.0,1)
      yy(1) = vtrans - 35.0
      call pstext(xx(1),yy(1),40,xlabel,8.0,1,0.0,1)
      xx(1) = htrans - 12.5
      yy(1) = vtrans + 0.5 * vgmax
      call pstext(xx(1),yy(1),40,ylabel,8.0,1,90.0,1)
c
c Label xmn, xmx, ymn, and ymx on plot:
c
      xx(1) = htrans
      yy(1) = vtrans - 25.0
      call numtext(hmn,title(1:12))
      call pstext(xx(1),yy(1),12,title,6.0,1,0.0,0)
      xx(1) = htrans + hgmax
      call numtext(hmx,title(1:12))
      call pstext(xx(1),yy(1),12,title,6.0,1,0.0,2)
      xx(1) = htrans - 5.0
      yy(1) = vtrans
      call numtext(vmn,title(1:12))
      call pstext(xx(1),yy(1),12,title,6.0,1,0.0,2)
      yy(1) = vtrans + vgmax - 20.0
      call numtext(vmx,title(1:12))
      call pstext(xx(1),yy(1),12,title,6.0,1,0.0,2)
c
c DRAW A LEGEND BAR:
c
      if(.not.skipleg) then
      
      write(lpsout,113)
 113  format('%',/,'% START OF LEGEND: ',/,'%')
c
c Specification of COLOR IMAGE for scale bar:
c
                    num  = 100
      if(icat.eq.1) num  = ncat
      ginc  = (cmax-cmin)/real(num)
      value = cmin - ginc
      do ih=1,num
            value = value + ginc
            if(icolor.eq.0.and.icat.eq.0) then
                  level = 256 - 256*(value-cmin)/(cmax-cmin)
                  level = max(min(level,255),0)
                  iline(ih) = hexa(level)
            else
                  if(icat.eq.0) then
                        value=min(cmax,max(cmin,value))
                        call red  (value,rline(ih),rfrac)
                        call green(value,gline(ih),gfrac)
                        call blue (value,bline(ih),bfrac)
                  else
                        jj        = ccode(ih)
                        rline(ih) = hexa(redint(jj))
                        gline(ih) = hexa(grnint(jj))
                        bline(ih) = hexa(bluint(jj))
                  end if
            end if
      end do
c
c Location of the Color Legend Bar.  Fix the X location and try
c to make the Y location about 80% of the height of the Y dimension:
c
      nh = num
      htrans = 930.0
      hscale =  90.0
      vtrans = 111.0
      vscale = 648.0
      if(icolor.eq.0.and.icat.eq.0) then
            write(lpsout,101)
            write(lpsout,'(200a2)') (iline(i),i=1,nh)
            write(lpsout,102) htrans,vtrans,hscale,vscale
            write(lpsout,103) 1,nh,1,nh
      else
            write(lpsout,105)
            write(lpsout,'(200a2)') (rline(i),i=1,nh)
            write(lpsout,106)
            write(lpsout,107)
            write(lpsout,'(200a2)') (gline(i),i=1,nh)
            write(lpsout,106)
            write(lpsout,108)
            write(lpsout,'(200a2)') (bline(i),i=1,nh)
            write(lpsout,106)
            write(lpsout,109) htrans,vtrans,hscale,vscale
            write(lpsout,110) 1,nh,1,nh
      endif
c
c Line around the legend:
c
      write(lpsout,114)
 114  format('%',/,'% BORDER AROUND LEGEND: ',/,'%',/,
     +       '0 0 0 setrgbcolor')
      thic = 0.3
      npts = 5
      xx(1) = htrans
      yy(1) = vtrans
      xx(2) = xx(1)
      yy(2) = vtrans + vscale
      xx(3) = htrans + hscale
      yy(3) = yy(2)
      xx(4) = xx(3)
      yy(4) = yy(1)
      xx(5) = xx(1)
      yy(5) = yy(1)
      call psline(npts,xx,yy,thic,0)
c
c Label the legend:
c
      tsiz = 4.0 + 3.0 * (vscale/672.0)
      xval = 1035.0
      if(icat.eq.1) then
            yinc = vscale / real(ncat)
            yval =(vtrans + (yinc-tsiz)/2.) - yinc
            do i=1,ncat
                  yval = yval + yinc
                  call pstext(xval,yval,12,cname(i),tsiz,1,0.0,0)
            end do
      else
            num   = int((cmax-cmin)/cinc+0.5)
            yinc  = vscale / real(num)
            cval  = cmin - cinc
            yval  = vtrans + 2.0 - yinc
            xx(1) = 1020.0
            xx(2) = 1030.0
            yy(1) = vtrans - yinc
            do i=1,num+1
                  cval = cval + cinc
                  if(ilog.eq.1) then
                        value = 10**cval
                  else
                        value = cval
                  endif
                  call numtext(value,title(1:12))
                  yval = yval + yinc
                  call pstext(xval,yval,12,title,tsiz,1,0.0,0)
                  yy(1) = yy(1) + yinc
                  yy(2) = yy(1)
                  call psline(2,xx,yy,thic,0)
            end do
      end if
c
c End of "if" loop over legend:
c
      end if
c
c Write Closing on PostScript File:
c
      write(lpsout,999)
 999  format('%END OF IMAGE',/,'4.166667 4.166667 scale',/,/,
     +       '%%EOF',/,'showpage')
c
c Finished:
c
      close(lpsout)
      write(*,9998) VERSION
 9998 format(/' PIXELPLT Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in parameters somewhere'
 99   write(*,*) 'At grid node: ',ix,iy,iz
      stop 'ERROR in data somewhere'
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
      open(lun,file='pixelplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for PIXELPLT',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/true.dat                 ',
     +       '-file with gridded data')
      write(lun,12)
 12   format('1                                ',
     +       '-  column number for variable')
      write(lun,13)
 13   format('-1.0e21  1.0e21                  ',
     +       '-  data trimming limits')
      write(lun,14)
 14   format('pixelplt.ps                      ',
     +       '-file with PostScript output')
      write(lun,15)
 15   format('1                                ',
     +       '-realization number')
      write(lun,16)
 16   format('50   0.5     1.0                 ',
     +       '-nx,xmn,xsiz')
      write(lun,17)
 17   format('50   0.5     1.0                 ',
     +       '-ny,ymn,ysiz')
      write(lun,18)
 18   format('1    0.0     1.0                 ',
     +       '-nz,zmn,zsiz')
      write(lun,19)
 19   format('1                                ',
     +       '-slice orientation: 1=XY, 2=XZ, 3=YZ')
      write(lun,20)
 20   format('1                                ',
     +       '-slice number     ')
      write(lun,21)
 21   format('2-D Reference Data               ',
     +       '-Title')
      write(lun,22)
 22   format('East                             ',
     +       '-X label')
      write(lun,23)
 23   format('North                            ',
     +       '-Y label')
      write(lun,24)
 24   format('0                                ',
     +       '-0=arithmetic, 1=log scaling')
      write(lun,25)
 25   format('1                                ',
     +       '-0=gray scale, 1=color scale')
      write(lun,26)
 26   format('0                                ',
     +       '-0=continuous, 1=categorical')
      write(lun,27)
 27   format('0.0  20.0  1.0                   ',
     +       '-continuous:  min, max, increm.')
      write(lun,28)
 28   format('4                                ',
     +       '-categorical: number of categories')
      write(lun,29)
 29   format('1     3    Code_One              ',
     +       '-category(), code(), name()')
      write(lun,30)
 30   format('2     1    Code_Two ',/,
     +       '3     6    Code_Three ')
      write(lun,31)
 31   format('4    10    Code_Four ',//)
      write(lun,32)
 32   format('Color Codes for Categorical Variable Plotting:')
      write(lun,33)
 33   format('      1=red, 2=orange, 3=yellow, 4=light green,',
     +       ' 5=green, 6=light blue,',/,
     +       '      7=dark blue, 8=violet, 9=white,',
     +       ' 10=black, 11=purple, 12=brown,',/,
     +       '      13=pink, 14=intermediate green, 15=gray')

      close(lun)
      return
      end
