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
c          Plot Arbitrary Number of PostScript Plots on Page
c          *************************************************
c
c
c output file - file containing all of the plots
c nx,ny       - number of plots in X and Y directions
c files       - files from this point forward will be used
c
c
c-----------------------------------------------------------------------
      use msflib
      
      parameter(MAXLEN=2048,MAXX=12,MAXY=12,VERSION=3.000)
      character str*2048,title*2048,pagenum*2048,
     +          psfile(MAXX*MAXY)*512,outfl*512
      logical   testfl
      data      lin/1/,lout/2/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' PLOTEM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'plotem.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'plotem.par          ') then
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

      read(lin,*,err=98) nx,ny
      write(*,*) ' number of plots in x and y = ',nx,ny
      ind = 0
      do iy=ny,1,-1
      do ix=1,nx
            ind = ind + 1
            psfile(ind)(1:6) = 'NOFILE'
      end do
      end do
      ind = 0
      do iy=ny,1,-1
      do ix=1,nx
            ind = ind + 1
            read(lin,'(a512)',err=2,end=2) psfile(ind)
            call chknam(psfile(ind),512)
            write(*,*) ' output file = ',psfile(ind)(1:40)
      end do
      end do
      read(lin,'(a)',err=2,end=2) title
      write(*,*) ' title = ',title(1:40)
      read(lin,'(a)',err=2,end=2) pagenum
      write(*,*) ' pagenum = ',title(1:40)
 2    continue
      write(*,*)
      close(lin)
c
c Prepare the output file:
c
      open(lout,file=outfl)
      write(lout,100)
 100  format('%!',//,'%',/,'% Output from PLOTEM',/,'%',//)
c
c Size and scaling parameters:
c
      dx = 310.
      dy = 220.
      xsiz = 504.0 / real(nx)
      ysiz = 684.0 / real(ny)
      if((xsiz/ysiz).gt.(dx/dy)) then
            xsiz = dx/dy*ysiz
      else
            ysiz = dy/dx*xsiz
      end if
      totx = xsiz * nx
      toty = ysiz * ny
      centx = (504.0-totx) / 2.0
      centy = (684.0-toty) / 2.0
      scalef = 0.9*xsiz/dx
      
      yorig = 78.0 + centy + real(ny)*ysiz
      call strlen(title,MAXLEN,lostr)
      write(lout,110) yorig,title(1:lostr)
 110  format('/ctext{ dup stringwidth pop -2 div 0 rmoveto show } def',
     +     /,'/TimesBold findfont 14 scalefont setfont', 
     +     /,'newpath 306 ',f8.2,' moveto (',a,') ctext')      
      yorig = 50.0
      call strlen(pagenum,MAXLEN,lostr)
      write(lout,111) yorig,pagenum(1:lostr)
 111  format('/Times findfont 11 scalefont setfont', 
     +     /,'newpath 306 ',f8.2,' moveto (',a,') ctext')      
c
c Loop over all input files:
c
      ind = 0
      do iy=ny,1,-1
      do ix=1,nx
      
      write(*,*) ' working on ix = ',ix,' iy = ',iy
      
      ind = ind + 1
      inquire(file=psfile(ind),exist=testfl)
      if(.not.testfl) go to 3
      
      yorig = 72.0 + centy + real(iy-1)*ysiz
      xorig = 72.0 + centx + real(ix-1)*xsiz
      write(lout,101) xorig,yorig,scalef,scalef
 101  format('%',/,'% PLOT',/,'%',/,'gsave ',2f12.3,' translate',
     +       2f8.4,' scale',/,/)
     
      open(lin,file=psfile(ind))      
      nsize = 0
 80   read(lin,*,end=90,err=99)
      nsize = nsize + 1
      go to 80
 90   continue
      rewind(lin)
      nsize = nsize -2
      do i = 1,11
            read(lin,*)
      end do
      do i = 12, nsize
            read(lin,'(a)',err=99)str
            call strlen(str,MAXLEN,lostr)
            write(lout,'(a)')str(1:lostr)
      end do
      close(lin)
      write(lout,102)
 102  format('grestore')

      
 3    continue

      end do
      end do

      write(lout,105)
 105  format(/,'showpage')
      
      close(lout)
      
      write(*,9998) VERSION
 9998 format(/' PLOTEM Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'ERROR in parameter file'
 99   stop 'ERROR in Postscript File.'
      end



      subroutine strlen(str,MAXLEN,lostr)
c-----------------------------------------------------------------------
c
c      Determine the length of the string minus trailing blanks
c
c
c
c-----------------------------------------------------------------------
      character str*2048
      lostr = MAXLEN
      do i=1,MAXLEN
            j = MAXLEN - i + 1
            if(str(j:j).ne.' ') return
            lostr = lostr - 1
      end do
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
      open(lun,file='plotem.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for PLOTEM',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('plotem.ps                     ',
     +       '-output file')
      write(lun,12)
 12   format('2  2                          ',
     +       '-number of plots in X and Y')
      write(lun,13)
 13   format('histplt.ps                    ',
     +       '-first plot file')
      write(lun,14)
 14   format('probplt.ps                    ',
     +       '-second plot file')
      write(lun,15)
 15   format('scatplt.ps                    ',
     +       '-third plot file')
      write(lun,16)
 16   format('vargplt.ps                    ',
     +       '-fourth plot file')

      close(lun)
      return
      end
