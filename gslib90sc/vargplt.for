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
c                          Plot Variograms
c                          ***************
c
c New version requires a parameter file (created if none given the first
c time the program is run)
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter (MAXLAG=5001,BIGNUM=1.0e21,EPSLON=1.0e-5,MAXCAT=24,
     +           VERSION=3.000)

      integer    redint(MAXCAT),grnint(MAXCAT),bluint(MAXCAT),test
      character  outfl*512,textfl*512,title*40,lotext(16)*80,str*512
      real       xx(MAXLAG),yy(MAXLAG)
      logical    testfl
c
c Declare dynamic arrays:
c
      character*512,allocatable :: datafl(:)
      real,allocatable          :: ar1(:,:),ar2(:,:)
      integer,allocatable       :: ivar(:),idash(:),ipts(:),iline(:),
     +                             nlag(:),iclr(:)
c
c Common variables for Postscript Plotting:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      common /olddat/ xvmn,xvmx,yvmn,yvmx
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
      data lin/1/,lpsout/2/,pscl/0.24/,pxmin/0.0/,pxmax/288.0/
     +     pymin/0.0/,pymax/216.0/,xmin/-10.0/,xmax/60.0/,
     +     ymin/-10.0/,ymax/60.0/,vpxmin/1.0/,vpxmax/59.5/,
     +     vpymin/0.0/,vpymax/58.0/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' VARGPLT Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'vargplt.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'vargplt.par         ') then
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
      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nvar
      write(*,*) ' number of variograms = ',nvar
c
c Read needed parameters:
c     
      MAXVAR = nvar
c
c Allocate the needed memory:
c1
      allocate(datafl(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c2
      allocate(ar1(MAXVAR,MAXLAG),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c3
      allocate(ar2(MAXVAR,MAXLAG),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c4
      allocate(ivar(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c5
      allocate(idash(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c6
      allocate(ipts(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c7
      allocate(iline(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c8
      allocate(nlag(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c9
      allocate(iclr(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  ' insufficient memory.'
                  stop
            end if
c           

      read(lin,*,err=98) dmin,dmax
      write(*,*) ' distance limits = ',dmin,dmax

      read(lin,*,err=98) gmin,gmax
      write(*,*) ' variogram limits = ',gmin,gmax

      read(lin,*,err=98) isill,sillval
      write(*,*) ' plot sill, sill value = ',isill,sillval

      read(lin,'(a40)') title
      call chktitle(title,40)
      write(*,*) ' title = ',title

      do i=1,nvar
            read(lin,'(a512)') datafl(i)
            call chknam(datafl(i),512)
            write(*,*) ' data file ',i,' = ',datafl(i)(1:40)
            read(lin,*) ivar(i),idash(i),ipts(i),iline(i),iclr(i)
            write(*,*) ' parameters ',
     +                  ivar(i),idash(i),ipts(i),iline(i),iclr(i)
      end do

      read(lin,'(a512)',err=95) textfl
      call chknam(textfl,512)
      write(*,*) ' text file ',textfl(1:40)
 95   continue

      close(lin)
c
c Read the variograms:
c
      do iii=1,nvar
c
c Find the right variogram:
c
      write(*,*) ' working on variogram ',iii
      write(*,*) ' trying to open       ',datafl(iii)(1:40)
      open(lin,file=datafl(iii))
      write(*,*) ' opened               ',datafl(iii)(1:40)
      read(lin,'(a80)') str
      do jjj=1,ivar(iii)-1
            do jj3=1,1000
                  read(lin,'(a80)') str
                  if(str(1:1).ne.' ') go to 333
            end do
 333        continue
      end do
c
c Read the variogram:
c
      np = 0
      do jjj=1,MAXLAG
            read(lin,'(a80)',end=334) str
            if(str(1:1).ne.' ') go to 334
            backspace lin
            np = np + 1
            read(lin,*,err=334,end=334) iiii,ar1(iii,np),ar2(iii,np)
            if(ar1(iii,np).lt.0.0001) np = np - 1
      end do
 334  continue
      nlag(iii) = np
      close(lin)
c
c Finish reading all variograms:
c
      end do
c
c Establish data-based limits to distance and variogram:
c
      dbdmin =  1.0e21
      dbdmax = -1.0e21
      dbvmin =  1.0e21
      dbvmax = -1.0e21
      do i=1,nvar
            do j=1,nlag(i)
                  if(ar1(i,j).lt.dbdmin) dbdmin = ar1(i,j)
                  if(ar1(i,j).gt.dbdmax) dbdmax = ar1(i,j)
                  if(ar2(i,j).lt.dbvmin) dbvmin = ar2(i,j)
                  if(ar2(i,j).gt.dbvmax) dbvmax = ar2(i,j)
            end do
      end do
      if(dmax.le.dmin) then
            dmin = 0.0
            dmax = dbdmax
      end if
      if(gmax.le.gmin) then
            gmin = 0.0
            gmax = dbvmax
            if(dbvmin.lt.0.0) gmin = dbvmin
      end if
      xvmin = dmin
      xvmax = dmax
      yvmin = gmin
      yvmax = gmax
c
c Write the variograms:
c
      open(lpsout,file=outfl,status='UNKNOWN')
      write(lpsout,998)
 998  format('%!PS                                 %    Remove     ',
     +    /, '90 234 translate 1.5 1.5 scale       %  these lines  ',
     +    /, '                                     % for EPSF file ',
     +    /, '%!PS-Adobe-3.0 EPSF-3.0',
     +    /, '%%BoundingBox: 0 0 288 216',
     +    /, '%%Creator: GSLIB',
     +    /, '%%Title:   ',
     +    /, '%%CreationDate: ',
     +    /, '%%EndComments',/,/,/,'%',/,'%',/,'%',/,
     +    /, '/m {moveto} def /l {lineto} def /r {rlineto} def',
     +    /, '/s {stroke} def /n {newpath} def /c {closepath} def',
     +    /, '/rtext{ dup stringwidth pop -1 div 0 rmoveto show } def',
     +    /, '/ctext{ dup stringwidth pop -2 div 0 rmoveto show } def',
     +    /, '/ltext{show} def /gr{grestore} def /gs{gsave} def',
     +    /, '/tr{translate} def /setc{setrgbcolor} def',
     +    /, '/bullet{ 8 0 360 arc c fill } def',/,/,
     +    /, '%72 72 translate',/,/,
     +    /, '0.240000 0.240000 scale')
c
c Draw and scale axes:
c
      ts   = 7.5
      xvmn = xvmin
      xvmx = xvmax
      yvmn = yvmin
      yvmx = yvmax
      write(lpsout,100)
 100  format('gsave /Symbol findfont 100 scalefont setfont ',
     +       '0 501 m (g) rtext grestore')
      xrange = vpxmax - vpxmin
      yrange = vpymax - vpymin
      xloc = vpxmin + 0.50*xrange
      yloc = vpymin - 0.15*yrange
      call pstext(xloc,yloc,8,'Distance',ts,1,0.0,1)
      xloc = vpxmin
      yloc = vpymax + 0.01*(vpxmax-vpxmin)
      call pstext(xloc,yloc,40,title,9.0,3,0.0,0)
      idsh = -1
      ilog = 0
      i45  = 0
      call scal(xvmin,xvmax,yvmin,yvmax,vpxmin,vpxmax,vpymin,vpymax,
     +          ilog,i45)
c
c Write variogram bullets and lines:
c
      do iii=1,nvar
c
c Color?
c
      if(iclr(iii).gt.0) then
            jclr = iclr(iii)
            irr  = redint(jclr)
            igg  = grnint(jclr)
            ibb  = bluint(jclr)
            write(lpsout,301) irr,igg,ibb
 301        format(3i4,' setrgbcolor')
      end if
c
c Mark bullets:
c
      if(ipts(iii).eq.1) then
            do i=1,nlag(iii)
                  xloc = resc(xvmn,xvmx,vpxmin,vpxmax,ar1(iii,i))
                  yloc = resc(yvmn,yvmx,vpymin,vpymax,ar2(iii,i))
                  if(ar1(iii,i).ge.dmin.and.ar1(iii,i).le.dmax.and.
     +               ar2(iii,i).ge.gmin.and.ar2(iii,i).le.gmax) then
                  ix = int((resc(xmin,xmax,pxmin,pxmax,xloc))/pscl)
                  iy = int((resc(ymin,ymax,pymin,pymax,yloc))/pscl)
                  write(lpsout,32) ix,iy
                  end if
            end do
 32         format('n ',i4.4,1x,i4.4,' bullet')
      end if
c
c Draw a line:
c
      if(iline(iii).eq.1) then
            np = 0
            do i=1,nlag(iii)
                  xx(i) = resc(xvmn,xvmx,vpxmin,vpxmax,ar1(iii,i))
                  yy(i) = resc(yvmn,yvmx,vpymin,vpymax,ar2(iii,i))
                  if(ar1(iii,i).lt.dmin.or.ar1(iii,i).gt.dmax.or.
     +               ar2(iii,i).lt.gmin.or.ar2(iii,i).gt.gmax) then
                        if(np.eq.0) np = i-1
                  end if
            end do
            idsh = idash(iii)
            np   = nlag(iii)
            call psline(np,xx,yy,0.5,idsh)
      end if
      write(lpsout,302)
 302  format('0 0 0 setrgbcolor')
c
c Finish loop over all annotation:
c
      end do
c
c Draw sill line?
c
      if(isill.eq.1) then
            np = 2
            xx(1) = resc(xvmn,xvmx,vpxmin,vpxmax,dmin)
            yy(1) = resc(yvmn,yvmx,vpymin,vpymax,sillval)
            xx(2) = resc(xvmn,xvmx,vpxmin,vpxmax,dmax)
            yy(2) = resc(yvmn,yvmx,vpymin,vpymax,sillval)
            call psline(np,xx,yy,0.5,0)
      end if
c
c Write the extra text file:
c
      inquire(file=textfl,exist=testfl)
      if(testfl) then
            write(*,*) ' trying to open ',textfl(1:40)
            open(1,file=textfl)
            nlot = 0
            do i=1,16
                  read(1,'(a80)',end=300) str(1:80)
                  nlot = nlot + 1
                  lotext(nlot) = str(1:80)
            end do
 300        continue
            xloc = vpxmin + 0.105*(vpxmax-vpxmin)
            yloc = vpymin - 0.010*(vpymax-vpymin)

            xl = (pxmin + 0.230*(pxmax-pxmin))/pscl
            xu = (pxmin + 0.995*(pxmax-pxmin))/pscl
            yl = (pymin + 0.150*(pymax-pymin))/pscl
            yu =  yl    +(real(nlot)*7.5    )/pscl
            write(lpsout,303) xl,yl,xl,yu,xu,yu,xu,yl
 303        format('n ',2f9.2,' m ',2f9.2,' l ',/,
     +             '  ',2f9.2,' l ',2f9.2,' l c ',/,
     +             'gs 1.0 1.0 0.7 setrgbcolor fill gr')

            ts = ts * 0.75
            do i=nlot,1,-1
                  yloc = yloc + ts*0.40
                  call pstext(xloc,yloc,72,lotext(i),ts,9,0.0,0)
            end do
      end if
c
c Close data file:
c
      write(lpsout,999)
 999  format('%END OF POSTSCRIPT FILE',/,'4.166667 4.166667 scale',/,/,
     +       '%%EOF',/,'showpage')
      close(lpsout)
c
c Finished:
c
 9997 write(*,9998) VERSION
 9998 format(/' VARGPLT Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'Error in parameter file'
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
      open(lun,file='vargplt.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for VARGPLT',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('vargplt.ps                   ',
     +       '-file for PostScript output')
      write(lun,12)
 12   format('2                            ',
     +       '-number of variograms to plot')
      write(lun,13)
 13   format('0.0   20.0                   ',
     +       '-distance  limits (from data if max<min)')
      write(lun,14)
 14   format('0.0    1.2                   ',
     +       '-variogram limits (from data if max<min)')
      write(lun,24)
 24   format('1      1.0                   ',
     +       '-plot sill (0=no,1=yes), sill value)')
      write(lun,15)
 15   format('Normal Scores Semivariogram  ',
     +       '-Title for variogram')
      write(lun,16)
 16   format('gamv.out                     ',
     +       '-1 file with variogram data')
      write(lun,17)
 17   format('1    1   1   1    1          ',
     +       '-  variogram #, dash #, pts?, line?, color')
      write(lun,18)
 18   format('vmodel.var                   ',
     +       '-2 file with variogram data')
      write(lun,19)
 19   format('1    3   0   1   10          ',
     +       '-  variogram #, dash #, pts?, line?, color')
      write(lun,20)
 20   format(//,'Color Codes for Variogram Lines/Points:')
      write(lun,21)
 21   format('      1=red, 2=orange, 3=yellow, 4=light green,',
     +       ' 5=green, 6=light blue,',/,
     +       '      7=dark blue, 8=violet, 9=white,',
     +       ' 10=black, 11=purple, 12=brown,',/,
     +       '      13=pink, 14=intermediate green, 15=gray')

      close(lun)
      return
      end
      subroutine chknam(str,len)
c-----------------------------------------------------------------------
c
c                   Check for a Valid File Name
c                   ***************************
c
c This subroutine takes the character string "str" of length "len" and
c removes all leading blanks and blanks out all characters after the
c first blank found in the string (leading blanks are removed first).
c
c
c
c-----------------------------------------------------------------------
      parameter (MAXLEN=512)
      character str(MAXLEN)*1
c
c Find first two blanks and blank out remaining characters:
c
      do i=1,len-1
            if(str(i)  .eq.' '.and.
     +         str(i+1).eq.' ') then
                  do j=i+1,len
                        str(j) = ' '
                  end do
                  go to 2
            end if
      end do
 2    continue
c
c Look for "-fi" for file
c
      do i=1,len-2
            if(str(i)  .eq.'-'.and.
     +         str(i+1).eq.'f'.and.
     +         str(i+2).eq.'i') then
                  do j=i+1,len
                        str(j) = ' '
                  end do
                  go to 3
            end if
      end do
 3    continue
c
c Look for "\fi" for file
c
      do i=1,len-2
            if(str(i)  .eq.'\'.and.
     +         str(i+1).eq.'f'.and.
     +         str(i+2).eq.'i') then
                  do j=i+1,len
                        str(j) = ' '
                  end do
                  go to 4
            end if
      end do
 4    continue
c
c Return with modified file name:
c
      return
      end
      subroutine chktitle(str,len)
c-----------------------------------------------------------------------
c
c                     Check for a Valid Title
c                     ***********************
c
c This subroutine takes the character string "str" of length "len" and
c blanks out all characters after the first back slash
c
c
c
c-----------------------------------------------------------------------
      parameter (MAXLEN=132)
      character str(MAXLEN)*1,strl*132
c
c Remove leading blanks:
c
      do i=1,len-1
            if(str(i).ne.' ') then
                  if(i.eq.1) go to 1
                  do j=1,len-i+1
                        k = j + i - 1
                        str(j) = str(k)
                  end do
                  do j=len,len-i+2,-1
                        str(j) = ' '
                  end do
                  go to 1
            end if
      end do
 1    continue
c
c Find first back slash and blank out the remaining characters:
c
      do i=1,len-1
            if(str(i).eq.'\\') then
                  do j=i,len
                        str(j) = ' '
                  end do
                  go to 2
            end if
      end do
 2    continue
c
c Although inefficient, copy the string:
c
      do i=1,len
            strl(i:i) = str(i)
      end do
c
c Look for a five character pattern with -Titl...
c
      do i=1,len-5
            if(strl(i:i+4).eq.'-Titl'.or.
     +         strl(i:i+4).eq.'-titl'.or.
     +         strl(i:i+4).eq.'-TITL'.or.
     +         strl(i:i+4).eq.'-X la'.or.
     +         strl(i:i+4).eq.'-Y la') then
                  do j=i,len
                        str(j) = ' '
                  end do
                  go to 3
            end if
      end do
 3    continue
c
c Return with modified character string:
c
      return
      end
      subroutine pstext(xs,ys,lostr,str,tsiz,ifont,rot,iadj)
c-----------------------------------------------------------------------
c
c              Write Postscript Text commands to a file
c              ****************************************
c
c
c CALLING ARGUMENTS:
c
c  xs            starting value of x in the range xmin to xmax
c  ys            starting value of y in the range ymin to ymax
c  lostr      number of characters in str to print
c  str            the character string
c  tsiz            Text size in 1/72 of an inch
c  ifont      Font Number: See font number below
c  rot             Rotation Angle to post the text (default to 0.0)
c  iadj            Adjustment: 0=left adjusted, 1=centre, 2=right
c
c
c-----------------------------------------------------------------------
      character str*80,fnnt(10)*32,line*132,size*4,part1*1,part2*7
c
c Common Block for Postscript Output Unit and Scaling:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      save    fnnt,ifold,tsold,izip
c
c Preset 10 different fonts:
c
      data fnnt/'/Helvetica             findfont ',
     +          '/Helvetica-Bold        findfont ',
     +          '/Helvetica-BoldOblique findfont ',
     +          '/Times-Roman           findfont ',
     +          '/Times-Bold            findfont ',
     +          '/Times-Italic          findfont ',
     +          '/Times-BoldItalic      findfont ',
     +          '/Courier               findfont ',
     +          '/CourierBold           findfont ',
     +          '/Courier-BoldOblique   findfont '/
      data ifold/0/,tsold/0.0/,izip/0/
      part1 = '('
      part2 = ')  text'
c
c Remove leading and trailing blanks:
c
      lost = lostr
 1    k = lost
      do i=1,k
            ix = k - i + 1
            if(str(ix:ix).ne.' ') go to 2
            lost = lost - 1
      end do
 2    if(lost.le.0) return
c
c Create line to set the text size and type:
c
      if(ifont.ne.ifold.or.tsiz.ne.tsold) then
            isiz=int(tsiz/pscl)
            write(size,'(i4)') isiz
            line=fnnt(ifont)//size//' scalefont setfont'      
            write(lpsout,'(a)')line(1:54)
            ifold = ifont
            tsold = tsiz
      endif
c
c Set the correct adjustment:
c
      part2(3:3) = 'l'
      if(iadj.eq.1) part2(3:3) = 'c'
      if(iadj.eq.2) part2(3:3) = 'r'
c
c Write the lines and position to the Postscript file:
c                  
      ix = int((resc(xmin,xmax,pxmin,pxmax,xs))/pscl)
      iy = int((resc(ymin,ymax,pymin,pymax,ys))/pscl)
c
c Rotate if Necessary:
c
      line = part1//str(1:lost)//part2
      if(rot.ne.0.0) then
            irot = int(rot)
            write(lpsout,102)   ix,iy
            write(lpsout,103)   irot 
            write(lpsout,100)   izip,izip
            write(lpsout,'(a)') line(1:lost+8)
            ix   = -1.0 * ix
            iy   = -1.0 * iy
            irot = -1.0 * irot
            write(lpsout,103)   irot 
            write(lpsout,102)   ix,iy
      else
c
c Just write out the text if no rotation:
c
            write(lpsout,100)   ix,iy
            write(lpsout,'(a)') line(1:lost+8)
      endif
 100  format(i5,1x,i5,1x,'m')
 102  format(i5,1x,i5,1x,'translate')
 103  format(i5,1x,'rotate')
c
c Finished - Return to calling program:
c
      return
      end
        subroutine scal(xmin,xmax,ymin,ymax,xaxmin,xaxmax,yaxmin,
     +                  yaxmax,ilog,i45)
c-----------------------------------------------------------------------
c
c Draws a reasonable graph axes for a PostScript plot.  The appropriate
c labelling and tic mark interval are established.
c
c INPUT VARIABLES:
c       xmin   - the minimum of the x axis (labeled on axis)
c       xmax   - the maximum of the x axis (labeled on axis)
c       ymin   - the minimum of the y axis (labeled on axis)
c       ymax   - the maximum of the y axis (labeled on axis)
c       xaxmin - the minimum of the x axis (on PostScript window)
c       xaxmax - the maximum of the x axis (on PostScript window)
c       yaxmin - the minimum of the y axis (on PostScript window)
c       yaxmax - the maximum of the y axis (on PostScript window)
c       ilog   - scale option: 0 - both cartesian
c                              1 - semi log with x being log scale
c                              2 - semi log with y being log scale
c                              3 - log log with both being log scale
c       i45    - 45 degree line option: 0 - no, 1 - if axes are the same
c              
c
c
c-----------------------------------------------------------------------
      parameter (EPS=0.001)
      real       xloc(5),yloc(5)
      character  label*8,lfmt*8
c
c Common Block for Postscript Output Unit and Scaling:
c
      common /psdata/ lpsout,psscl,pxmin,pxmax,pymin,pymax,wxmin,
     +                wxmax,wymin,wymax
c
c Check to make sure that the scale can be plotted:
c
      if((xmax-xmin).le.0.0001.or.(ymax-ymin).le.0.0001) return
c
c Set up some of the parameters:
c
      tlng =       0.013 * ((xaxmax-xaxmin) + (yaxmax-yaxmin))
      tsht =       0.007 * ((xaxmax-xaxmin) + (yaxmax-yaxmin))
      psz  = 4.0 + 0.060 *  (xaxmax-xaxmin)
      pl1  = 0.6 + 0.005 *  (xaxmax-xaxmin)
      pl2  = 0.3 + 0.003 *  (xaxmax-xaxmin)
c
c Draw the axis:
c
      xloc(1) = xaxmin
      yloc(1) = yaxmax
      xloc(2) = xaxmin
      yloc(2) = yaxmin
      xloc(3) = xaxmax
      yloc(3) = yaxmin
      call psline(3,xloc,yloc,pl1,0)
c
c Show a 45 degree line?
c
      if(i45.eq.1) then
         if(abs(xmin-ymin).le.0.0001.and.abs(xmax-ymax).le.0.0001) then
            xloc(1) = xaxmin
            yloc(1) = yaxmin
            xloc(2) = xaxmax
            yloc(2) = yaxmax
            call psline(2,xloc,yloc,pl1,0)
         end if
      end if
c
c CONSTRUCT THE X AXIS:
c
c
c Log scale?
c
      if(ilog.eq.1.or.ilog.eq.3) then
c
c      The start, end, number of log(10) cycles, the tic mark start,
c      and the number of points in defining a tic mark:
c
            tminx   = alog10(xmin)
            tmaxx   = alog10(xmax)
            ncyc    = tmaxx - tminx
            cbas    = xmin/10
            yloc(1) = yaxmin
            num     = 2
c
c      Loop along the axis drawing the tic marks and labels:
c
            do icyc=1,ncyc+1
                  cbas = cbas * 10
                  do i=1,9
                  t1   = alog10(cbas*real(i))
                  xloc(1) = resc(tminx,tmaxx,xaxmin,xaxmax,t1)
                  xloc(2) = xloc(1)
                  if(i.eq.1) then
c
c            First point - long tic mark:
c
                        yloc(2) = yloc(1) - tlng
                        call psline(num,xloc,yloc,pl2,0)
                        yloc(2) = yloc(1) - 2.5*tlng
                        if(abs(t1+9.).le.EPS) label = '1.0e-9  '
                        if(abs(t1+8.).le.EPS) label = '1.0e-8  '
                        if(abs(t1+7.).le.EPS) label = '1.0e-7  '
                        if(abs(t1+6.).le.EPS) label = '1.0e-6  '
                        if(abs(t1+5.).le.EPS) label = '0.00001 '
                        if(abs(t1+4.).le.EPS) label = '0.0001  '
                        if(abs(t1+3.).le.EPS) label = '0.001   '
                        if(abs(t1+2.).le.EPS) label = '0.01    '
                        if(abs(t1+1.).le.EPS) label = '0.1     '
                        if(abs(t1)   .le.EPS) label = '1       '
                        if(abs(t1-1.).le.EPS) label = '10      '
                        if(abs(t1-2.).le.EPS) label = '100     '
                        if(abs(t1-3.).le.EPS) label = '1000    '
                        if(abs(t1-4.).le.EPS) label = '10000   '
                        if(abs(t1-5.).le.EPS) label = '100000  '
                        if(abs(t1-6.).le.EPS) label = '1.0e+6  '
                        if(abs(t1-7.).le.EPS) label = '1.0e+7  '
                        if(abs(t1-8.).le.EPS) label = '1.0e+8  '
                        if(abs(t1-9.).le.EPS) label = '1.0e+9  '
                        call pstext(xloc(1),yloc(2),8,label,
     +                                      psz,1,0.0,1)
                  else
c
c            Not first point - short tic mark:
c
                        if(icyc.le.ncyc) then
                              yloc(2) = yloc(1) - tsht
                              call psline(num,xloc,yloc,pl2,0)
                        endif
                  endif
                    end do
              end do
      else
c
c Arithmetic Scale:
c
            do i=1,20
                  test = (xmax-xmin)/(10.0**(6-i))
                  if(test.gt.0.9) go to 1
            end do
 1          if(test.gt.3.0)                 zval = 1.0
            if(test.le.3.0.and.test.gt.2.0) zval = 0.5
            if(test.le.2.0.and.test.gt.1.2) zval = 0.4
            if(test.le.1.2)                 zval = 0.2
            nval = 5
            if(zval.eq.0.4.or.zval.eq.0.2)  nval = 4
            zval = zval * 10.0**(6-i)
            tval = zval / real(nval)
            if(i.ge.12) lfmt = '(f8.8)'
            if(i.eq.11) lfmt = '(f8.7)'
            if(i.eq.10) lfmt = '(f8.6)'
            if(i.eq.9)  lfmt = '(f8.5)'
            if(i.eq.8)  lfmt = '(f8.4)'
            if(i.eq.7)  lfmt = '(f8.3)'
            if(i.eq.6)  lfmt = '(f8.2)'
            if(i.eq.5)  lfmt = '(f8.1)'
            if(i.le.4)  lfmt = '(f8.0)'
c
c      Loop along the axis drawing the tic marks and labels:
c
            yloc(1) = yaxmin
            pos     = xmin
            num     = 2
            do i=1,100
                  yloc(2) = yaxmin - tlng
                  xloc(1) = resc(xmin,xmax,xaxmin,xaxmax,pos)
                  xloc(2) = resc(xmin,xmax,xaxmin,xaxmax,pos)
                  call psline(num,xloc,yloc,pl2,0)
                  yloc(2) = yloc(1) - 2.5*tlng
                  write(label,lfmt) pos                        
                  call pstext(xloc(1),yloc(2),8,label,psz,1,0.0,1)
                  yloc(2) = yaxmin - tsht
                  do j=1,nval-1
                       pos     = pos + tval
                       if(pos.gt.xmax) go to 2
                       xloc(1) = resc(xmin,xmax,xaxmin,xaxmax,pos)
                       xloc(2) = resc(xmin,xmax,xaxmin,xaxmax,pos)
                       call psline(num,xloc,yloc,pl2,0)
                    end do
                  pos = pos + tval
                  if(pos.gt.xmax) go to 2
            end do
 2          continue
      endif
c
c CONSTRUCT THE Y AXIS:
c
c
c Log scale?
c
      if(ilog.eq.2.or.ilog.eq.3) then
c
c      The start, end, number of log(10) cycles, the tic mark start,
c      and the number of points in defining a tic mark:
c
            tminy   = alog10(ymin)
            tmaxy   = alog10(ymax)
            ncyc    = tmaxy - tminy
            cbas    = ymin/10
            xloc(1) = xaxmin
            num     = 2
c
c      Loop along the axis drawing the tic marks and labels:
c
            do icyc=1,ncyc+1
                  cbas = cbas * 10
                  do i=1,9
                  t1   = alog10(cbas*real(i))
                  yloc(1) = resc(tminy,tmaxy,yaxmin,yaxmax,t1)
                  yloc(2) = yloc(1)
                  if(i.eq.1) then
c
c            First point - long tic mark:
c
                        xloc(2) = xloc(1) - tlng
                        call psline(num,xloc,yloc,pl2,0)
                        xloc(2) = xloc(2) - 0.1*tlng
                        if(abs(t1+9.).le.EPS) label = '1.0e-9  '
                        if(abs(t1+8.).le.EPS) label = '1.0e-8  '
                        if(abs(t1+7.).le.EPS) label = '1.0e-7  '
                        if(abs(t1+6.).le.EPS) label = '1.0e-6  '
                        if(abs(t1+5.).le.EPS) label = '0.00001 '
                        if(abs(t1+4.).le.EPS) label = '0.0001  '
                        if(abs(t1+3.).le.EPS) label = '0.001   '
                        if(abs(t1+2.).le.EPS) label = '0.01    '
                        if(abs(t1+1.).le.EPS) label = '0.1     '
                        if(abs(t1)   .le.EPS) label = '1       '
                        if(abs(t1-1.).le.EPS) label = '10      '
                        if(abs(t1-2.).le.EPS) label = '100     '
                        if(abs(t1-3.).le.EPS) label = '1000    '
                        if(abs(t1-4.).le.EPS) label = '10000   '
                        if(abs(t1-5.).le.EPS) label = '100000  '
                        if(abs(t1-6.).le.EPS) label = '1.0e+6  '
                        if(abs(t1-7.).le.EPS) label = '1.0e+7  '
                        if(abs(t1-8.).le.EPS) label = '1.0e+8  '
                        if(abs(t1-9.).le.EPS) label = '1.0e+9  '
                        call pstext(xloc(2),yloc(2),8,label,
     +                                      psz,1,0.0,2)
                  else
c
c            Not first point - short tic mark:
c
                        if(icyc.le.ncyc) then
                              xloc(2) = xloc(1) - tsht
                              call psline(num,xloc,yloc,pl2,0)
                        endif
                  endif
                  end do
            end do
      else
c
c      Determine a labelling and tic mark increment:
c
            do i=1,20
                  test = (ymax-ymin)/(10.0**(6-i))
                  if(test.gt.0.9) go to 11
            end do
 11         if(test.ge.3.0)                 zval = 1.0
            if(test.le.3.0.and.test.gt.2.0) zval = 0.5
            if(test.le.2.0.and.test.gt.1.2) zval = 0.4
            if(test.le.1.2)                 zval = 0.2
            nval = 5
            if(zval.eq.0.4.or.zval.eq.0.2)  nval = 4
            zval = zval * 10.0**(6-i)
            tval = zval / real(nval)
            if(i.ge.12) lfmt = '(f8.8)'
            if(i.eq.11) lfmt = '(f8.7)'
            if(i.eq.10) lfmt = '(f8.6)'
            if(i.eq.9)  lfmt = '(f8.5)'
            if(i.eq.8)  lfmt = '(f8.4)'
            if(i.eq.7)  lfmt = '(f8.3)'
            if(i.eq.6)  lfmt = '(f8.2)'
            if(i.eq.5)  lfmt = '(f8.1)'
            if(i.le.4)  lfmt = '(f8.0)'
c
c      Loop along the axis drawing the tic marks and labels:
c
            xloc(1) = xaxmin
            pos     = ymin
            num     = 2
            do i=1,100
                  xloc(2) = xaxmin - tlng
                  yloc(1) = resc(ymin,ymax,yaxmin,yaxmax,pos)
                  yloc(2) = resc(ymin,ymax,yaxmin,yaxmax,pos)
                  call psline(num,xloc,yloc,pl2,0)
                  xloc(2) = xloc(2) - 0.2*tlng
                  write(label,lfmt) pos                        
                  call pstext(xloc(2),yloc(2),8,label,psz,1,0.0,2)
                  xloc(2) = xaxmin - tsht
                  do j=1,nval-1
                       pos     = pos + tval
                       if(pos.gt.ymax) go to 12
                       yloc(1) = resc(ymin,ymax,yaxmin,yaxmax,pos)
                       yloc(2) = resc(ymin,ymax,yaxmin,yaxmax,pos)
                       call psline(num,xloc,yloc,pl2,0)
                     end do
                  pos = pos + tval
                  if(pos.gt.ymax) go to 12
            end do
 12         continue
      endif
c
c Return to calling program:
c
      return
      end
      real function resc(xmin1,xmax1,xmin2,xmax2,x111)
      real*8 rsc
c
c Simple linear rescaling (get a value in coordinate system "2" given
c a value in "1"):
c
      rsc  = dble((xmax2-xmin2)/(xmax1-xmin1))
      resc = xmin2 + real( dble(x111 - xmin1) * rsc )
      return
      end
      subroutine psline(np,x,y,lwidt,idsh)
c-----------------------------------------------------------------------
c
c              Write Postscript line commands to a file
c              ****************************************
c
c
c CALLING ARGUMENTS:
c
c  np           the number of points in the x and y array to join
c  x()          array of x values in the range xmin to xmax
c  y()          array of y values in the range ymin to ymax
c  lwidt        the width of the line (1.0 = dark, 0.5 = light)
c  idsh         Dashing Index
c
c NOTES:
c
c  1. The pxmin,pxmax,.. variables are in the standard 1/72 inch 
c     resolution of the postscript page. If a different scale is 
c     going to be used in the printing set pscl to the scale.
c
c  2. If "idsh" is zero then no dashing is perfomed
c
c
c-----------------------------------------------------------------------
      real      x(*),y(*),lwidt,lwold
      character dash(10)*24
c
c Common Block for Postscript Output Unit and Scaling:
c
      common /psdata/ lpsout,pscl,pxmin,pxmax,pymin,pymax,xmin,
     +                xmax,ymin,ymax
      save   lwold
c
c Dash Patterns:
c
      data dash/'[40 20] 0 setdash       ',
     +          '[13 14 13 20] 0 setdash ',
     +          '[12 21 4 21] 0 setdash  ',
     +          '[10 10] 0 setdash       ',
     +          '[20 20] 0 setdash       ',
     +          '[30 30] 0 setdash       ',
     +          '[40 40] 0 setdash       ',
     +          '[50 50] 0 setdash       ',
     +          '[50 50] 0 setdash       ',
     +          '[50 50] 0 setdash       '/
c
c Change the line width if necessary:
c
      if(pscl.lt.0.01) pscl = 1.0
      if(idsh.gt.10)   idsh = 10 
      if(lwidt.ne.lwold) then
            width = lwidt/pscl
            write(lpsout,100) width
 100        format(f6.3,' setlinewidth')
            lwold = lwidt
      endif
c
c Start a new path and loop through the points:
c
      if(idsh.gt.0) write(lpsout,'(a24)') dash(idsh)
      write(lpsout,101)
 101  format('n')
      do i=1,np
            ix = int(resc(xmin,xmax,pxmin,pxmax,x(i))/pscl)
            iy = int(resc(ymin,ymax,pymin,pymax,y(i))/pscl)
            if(i.eq.1) then
                  write(lpsout,102) ix,iy
 102              format(i5,1x,i5,' m')
            else
                  write(lpsout,103) ix,iy
 103              format(i5,1x,i5,' l')
            endif
      end do      
      write(lpsout,104)
 104  format('s')
      if(idsh.gt.0) write(lpsout,105)
 105  format('[] 0 setdash')
c
c Finished - Return to calling program:
c
      return
      end
