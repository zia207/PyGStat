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
c
c Module to declare dynamic arrays in multiple subroutines:
c
      module geostat
      
      real,allocatable    :: prop(:)
      integer,allocatable :: var(:,:,:),ntrain(:,:,:,:),nact(:,:,:,:),
     +                       ntry(:,:,:,:,:),num(:,:),ixl(:),iyl(:),
     +                       izl(:),nlag(:),icross(:),icat(:)

      real      VERSION,tol
      integer   test,report,nsim,nx,ny,nz,nr,np,nv,lout,idbg,ldbg,
     +          maxit,ndir,mlag,maxroc
      character datafl*512,trainfl*512,outfl*512,dbgfl*512

      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c              Post Process an Integer Coded Realization
c              *****************************************
c
c Input image(s) are post-processed with a steepest descent approach
c so that the two-point histogram for specified directions and lags are
c identified to that of a training image.
c
c
c
c INPUT/OUTPUT Variables:
c
c    datafl     file containing realizations to be post-processed
c    trainfl    file containing the training image
c    outfl      output file for post-processed realizations
c    idbg       debugging level (0-3)
c    report     after ``report'' loops the image is written to the
c               output file
c    dbgfl      output file for debugging information
c    maxit      maximum number of iterations over all nodes
c    tol        tolerance for the objective function (starts at 1)
c    nsim       number of simulations to post-process
c    nx,ny,nz   grid size for the input realizations AND training image
c    ndir       the number of directions to consider for the two-point
c               histogram control
c    ixl(),iyl(),izl()  the offsets for each direction
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use      geostat
      VERSION = 3.000
c
c Read the Parameter File, training image, and initialize random image:
c
      call readparm
c
c LOOP over all the simulations:
c
      do isim=1,nsim
            call initgrid
            call tphist(1)
            call sim
            call tphist(1)
      end do
c
c Finished:
c
      close(lout)
      close(ldbg)
      write(*,9998) VERSION
 9998 format(/' ANNEAL Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
 
      subroutine readparm
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters are read from a file name provided from standard
c input (a default name will be tried if none is keyed in by the user).
c
c
c
c-----------------------------------------------------------------------
      use       msflib
      use       geostat
c
c ACORN parameters:
c
      parameter   (KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
      common /iaco/   ixv(MAXOP1)
      real*8    acorni

      character title*80,str*512
      logical   testfl
      lin  = 1
      lout = 2
      ldbg = 3
      linp = 10
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' ANNEAL Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'anneal.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'anneal.par          ') then
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
      write(*,*) ' input images = ',datafl(1:40)

      read(lin,'(a512)',err=97) trainfl
      call chknam(trainfl,512)
      write(*,*) ' training image = ',trainfl(1:40)

      read(lin,*,err=97) idbg,report
      write(*,*) ' idbg, report = ',idbg,report

      read(lin,'(a512)',err=97) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debug file = ',dbgfl(1:40)

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output images = ',outfl(1:40)

      read(lin,*,err=97) nsim
      write(*,*) ' # of simulations = ',nsim

      read(lin,*,err=97) nx,ny,nz
      write(*,*) ' nx,ny,nz = ',nx,ny,nz

      read(lin,*,err=97) ixv(1)
      write(*,*) ' random number seed',ixv(1)

      read(lin,*,err=97) maxit,tol
      write(*,*) ' maxit,tol = ',maxit,tol

      read(lin,*,err=97) ndir
      write(*,*) ' ndir = ',ndir

      allocate (ixl(ndir),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (iyl(ndir),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (izl(ndir),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (nlag(ndir),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (icross(ndir),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      

      mlag = 0
      do i=1,ndir
            read(lin,*,err=97)  ixl(i),iyl(i),izl(i),
     +                              nlag(i)
            icross(i) = 1
            if(nlag(i).gt.mlag) mlag = nlag(i)
            write(*,120) i,ixl(i),iyl(i),izl(i),nlag(i),icross(i)
      end do
      close(lin)
 120  format(' Direction',i2,' ixl iyl izl: ',3i3,' nlag: ',i3,
     +       ' icross: ',i2)
c
c ACORN Initialization (if the seed is chosen small then the first few
c                       random numbers are always small - loop KORDEI)
c
      do i=2,KORDEI+1
            ixv(i) = 0
      end do
      do i=1,KORDEI*KORDEI
            randnu = real(acorni(idum))
      end do
c
c Open the debugging and output files:
c
      open(ldbg,file=dbgfl,status='UNKNOWN')
      open(lout,file=outfl,status='UNKNOWN')
      title = 'ANNEAL SIMULATIONS:                      '//
     +        '                                         '
      inquire(file=trainfl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR file ',trainfl(1:40),' does not exist!'
            write(*,*) '  you need a training image'
            stop
      end if
c
c Allocate the needed memory:
c
      allocate (var(nx,ny,nz),stat = test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
c Read through training data file:
c
      ncat = 0
      open(lin,file=trainfl,status='UNKNOWN')
      read(lin,*,err=98)
      read(lin,*,err=98) nvari
      do i=1,nvari
            read(lin,*,err=98)
      end do
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        read(lin,*,err=98) tempvar
                        do i=1,ncat
                              if(tempvar.eq.var(i,1,1)) go to 71
                        end do
                        ncat = ncat + 1
                        var(i,1,1) = tempvar
 71                     continue
                  end do
            end do
      end do
      close(lin)
c
c Finish allocation:
c
      allocate (prop(ncat+1),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if     
      allocate (ntrain(ncat+1,ncat+1,ndir,mlag),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if    
      allocate (nact(ncat+1,ncat+1,ndir,mlag),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if    
      allocate (ntry(ncat,ncat+1,ncat+1,ndir,mlag),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (num(ndir,mlag),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (icat(ncat+1),stat=test)
      if(test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
c
c Initialize the grid:
c
      maxroc = ncat + 1
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        var(ix,iy,iz) = maxroc
                  end do
            end do
      end do
c
c Open the file with the input images and read off the header:
c
      open(linp,file=datafl,status='UNKNOWN')
      read(linp,*)
      read(linp,*) nvari
      do i=1,nvari
            read(linp,*)
      end do

c
c Read through training data file:
c
      ncat = 0
      open(lin,file=trainfl,status='UNKNOWN')
      read(lin,'(a60)',err=98) title(21:80)
      read(lin,*,err=98) nvari
      do i=1,nvari
            read(lin,*,err=98)
      end do
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        read(lin,*,err=98) tempvar
                        icode = int(tempvar+0.5)
                        do i=1,ncat
                              if(icode.eq.icat(i)) go to 72
                        end do
                        ncat = ncat + 1
                        icat(ncat)    = icode
 72                     var(ix,iy,iz) = icode
                  end do
            end do
      end do
      close(lin)
c
c Initialize the total proportion and the missing value code:
c
      do i = 1,maxroc
            prop(i) = 0.0
      end do
      tprop   = 0.0
      imiss   = 1
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        ind   = var(ix,iy,iz)
                        icode = imiss
                        do i=1,ncat
                              if(ind.eq.icat(i)) icode = i
                        end do
                        if(icode.ge.1.and.icode.le.maxroc) then
                              prop(icode) = prop(icode) + 1.0
                              tprop       = tprop + 1
                        endif
                        var(ix,iy,iz) = icode
                  end do
            end do
      end do
c
c Number of rock types, compute two point histogram of training image:
c
      nr   = ncat
      nxyz = nx * ny * nz
      write(*,100) nr,(i,icat(i),prop(i),i=1,nr)
 100  format('There are ',i2,' rock types:',/,10
     +        ('   ',i2,' code: ',i3,' there are : ',f10.0,/))
      tprop = 1.0 / max(tprop,1.0)
      oldpro = 0.0
      do i=1,nr
            prop(i) = oldpro + prop(i)*tprop
            oldpro  = prop(i)
      end do
      call tphist(0)
c
c Write a header on the output file and return:
c
      write(lout,101) title
 101  format(a80)

      xmn   = 0.
      ymn   = 0.
      zmn   = 0.
      xsiz  = 1.
      ysiz  = 1.
      zsiz  = 1. 
      nreal = 1     
      write(lout,102) nvar,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nreal
 102  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,103)
 103  format('simulation')

      return
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in distribution file!'
      end
 
 
 
      subroutine initgrid
c-----------------------------------------------------------------------
c
c                      Initialization of Grid
c                      **********************
c
c
c
c-----------------------------------------------------------------------
      use      geostat
      data linp/10/
c
c Read in a new initial image:
c
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        read(linp,*) tempvar
                        k = int(tempvar+0.5)
                        do i=1,nr
                              if(k.eq.icat(i)) icode = i
                        end do
                        var(ix,iy,iz) = icode
                  end do
            end do
      end do
      return
      end
 
 
 
      subroutine sim
c-----------------------------------------------------------------------
c
c               MAIN ROUTINE TO PERFORM THE SIMULATION
c
c
c
c
c-----------------------------------------------------------------------
      use      geostat
      real*8   acorni
c
c ACORN parameters:
c
      parameter   (KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
      common /iaco/   ixv(MAXOP1)
c
c Initialize the starting two point histogram, and objective funtion:
c
      current = object(0)
      objrsc  = 1.0/max(current,0.000001)
      current = objrsc*current
c
c The random path generation procedure of Srivastava:
c
      nxy  = nx * ny
      nxyz = nx * ny * nz
      modlus = 1
 1    modlus = modlus * 2
      if(modlus.le.nxyz) go to 1
c
c MAIN LOOP TO MAXIMUM NUMBER OF ITERATIONS:
c
      do 2 iloop=1,maxit
            write(*,100) iloop,current
 100        format(' Working on ',i2,' objective function: ',f12.9)
            if(current.lt.tol) go to 31
c
c MAIN LOOP OVER ALL THE NODES (unless we're done?):
c
            index = int(real(acorni(idum))*nxyz+1)
            do 3 in=1,nxyz
c
c Figure out the location of this node:
c
            iz = int((index-1)/nxy) + 1
            iy = int((index-(iz-1)*nxy-1)/nx) + 1
            ix = index - (iz-1)*nxy - (iy-1)*nx
c
c Compute the two point histogram and objective function for rock types:
c
            call update(ix,iy,iz)
            best  = current
            ibest = var(ix,iy,iz)
            do ir=1,nr
                  if(ir.ne.var(ix,iy,iz)) then
                        objtry = objrsc*object(ir)
                        if(objtry.lt.best) then
                              best  = objtry
                              ibest = ir
                        endif
                  endif
            end do
c
c Decide on the best and keep it:
c            
            if(ibest.ne.var(ix,iy,iz)) then
                  var(ix,iy,iz) = ibest
                  current = best
                  do ir=1,maxroc
                     do jr=1,maxroc
                        do id=1,ndir
                           do il=1,nlag(id)
                            nact(ir,jr,id,il) = ntry(ibest,ir,jr,id,il)
                           end do
                        end do
                     end do
                  end do
            endif
c
c Get a new node to consider:
c
 20         index = mod(5*index+1,modlus)
            if(index.gt.nxyz.or.index.lt.1) go to 20
c
c END MAIN LOOP OVER NODES:
c
 3          continue
c
c Write this realization to the output file?
c
 31         if((int(iloop/report)*report).eq.iloop.or.
     +            iloop.eq.maxit.or.current.lt.tol) then
                  do iz=1,nz
                        do iy=1,ny
                              do ix=1,nx
                                    ind = var(ix,iy,iz)
                                    write(lout,'(i1)')  icat(ind)
                              end do
                        end do
                  end do
                  if(current.lt.tol) return
            endif
c
c END MAIN LOOP OVER MAXIMUM ITERATIONS:
c
 2    continue
      return
      end
 
 
      subroutine update(ix,iy,iz)
c-----------------------------------------------------------------------
c
c       Considering a node - decide on the best value to keep
c
c
c
c
c-----------------------------------------------------------------------
      use      geostat
c
      inow = var(ix,iy,iz)
c
c Initialize:
c
      do irtry=1,nr
            do ir=1,maxroc
               do jr=1,maxroc
                  do id=1,ndir
                     do il=1,nlag(id)
                        ntry(irtry,ir,jr,id,il) = nact(ir,jr,id,il)
                     end do
                  end do
               end do
            end do
      end do
c
c MAIN LOOP OVER ALL ROCK TYPES:
c
      do 10 ir=1,nr
            if(ir.eq.inow) go to 10
c
c Consider the change to all lags and directions:
c
            do 3 id=1,ndir
            do 4 il=1,nlag(id)

c
c      Update the positive lag:
c
            jx = ix + il*ixl(id)
            jy = iy + il*iyl(id)
            jz = iz + il*izl(id)
            if(jx.gt.nx.or.jy.gt.ny.or.jz.gt.nz) go to 40
            if(jx.lt. 1.or.jy.lt. 1.or.jz.lt. 1) go to 40
            j  = var(jx,jy,jz)
            ntry(ir,inow,j,id,il) = ntry(ir,inow,j,id,il) - 1
            ntry(ir,ir,  j,id,il) = ntry(ir,ir,  j,id,il) + 1
c
c      Update the negative lag:
c
 40         jx = ix - il*ixl(id)
            jy = iy - il*iyl(id)
            jz = iz - il*izl(id)
            if(jx.gt.nx.or.jy.gt.ny.or.jz.gt.nz) go to 4
            if(jx.lt. 1.or.jy.lt. 1.or.jz.lt. 1) go to 4
            j  = var(jx,jy,jz)
            ntry(ir,j,inow,id,il) = ntry(ir,j,inow,id,il) - 1
            ntry(ir,j,ir,  id,il) = ntry(ir,j,ir,  id,il) + 1

 4          continue
 3          continue
c
c END loop over rock types:
c
 10   continue
      return
      end
 
 
 
      subroutine tphist(iflag)
c-----------------------------------------------------------------------
c
c Compute a two point histogram of an array "var" that is nx by ny by nz
c (dimensioned for MAXX by MAXY by MAXZ), for "nr" rock types, "nlag"
c lags, "ndir" directions specified by integer arrays "ixl", "iyl",
c "izl".  Put the result in the "tp" array that is dimensioned MP by MV
c
c
c-----------------------------------------------------------------------
      use      geostat
c
c Initialize:
c
      do ir=1,maxroc
         do jr=1,maxroc
            do id=1,ndir
               do il=1,nlag(id)
                  if(iflag.eq.0) then
                        ntrain(ir,jr,id,il) = 0
                  else
                        nact  (ir,jr,id,il) = 0
                  endif
                  num(id,il) = 0
               end do
            end do
         end do
      end do
c
c Calculate the Experimental two point histogram:
c
      do 2 iz=1,nz
      do 2 iy=1,ny
      do 2 ix=1,nx
c
c Consider the first value and all directions and lags:
c
            i = var(ix,iy,iz)
            do 3 id=1,ndir
            do 3 il=1,nlag(id)
                  ii = ix + il*ixl(id)
                  jj = iy + il*iyl(id)
                  kk = iz + il*izl(id)
                  if(ii.gt.nx.or.jj.gt.ny.or.kk.gt.nz) go to 3
                  if(ii.lt. 1.or.jj.lt. 1.or.kk.lt. 1) go to 3
c
c If the second point is within the grid then keep the sample:
c
                  j = var(ii,jj,kk)
                  if(iflag.eq.0) then
                        ntrain(i,j,id,il) = ntrain(i,j,id,il)+1
                  else
                        nact  (i,j,id,il) = nact  (i,j,id,il)+1
                  endif
                  num(id,il) = num(id,il) + 1
 3          continue
 2    continue
c
c Debugging output:
c
      if(idbg.ge.3) then
            do ir=1,nr
            do jr=1,nr
            do id=1,ndir
            do il=1,nlag(id)
                  write(ldbg,100) ir,jr,id,il,ntrain(ir,jr,id,il),
     +                                        nact  (ir,jr,id,il)
            end do
            end do
            end do
            end do
 100        format('ir jr ',2i5,' id il ',2i5,' train act ',2i12)
      endif
c
c Return with the two point histogram:
c
      return
      end



      real function object(iflag)
c-----------------------------------------------------------------------
c
c Compute the objective function given a training two point histogram
c and an experimental two point histogram.
c
c
c-----------------------------------------------------------------------
      use      geostat
      object = 0.0
c
c Loop over all direction vectors and rock codes:
c
      do 1 id=1,ndir
      do 1 il=1,nlag(id)
            if(num(id,il).gt.0) then
                  do 2 ir=1,nr
                  do 2 jr=1,nr
                  if(icross(id).eq.0.and.ir.ne.jr) go to 2
                  if(iflag.eq.0) then
                        actual = nact(ir,jr,id,il)
                  else
                        actual = ntry(iflag,ir,jr,id,il)
                  endif
                  object = object + (actual-ntrain(ir,jr,id,il))
     +                                  * (actual-ntrain(ir,jr,id,il))
     +                                  /    (num(id,il)*num(id,il))
 2                continue
            endif
 1    continue
c
c Return with the objective function:
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
      open(lun,file='anneal.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for ANNEAL',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('sisim.out                         ',
     +       '-file with input image(s)')
      write(lun,12)
 12   format('ellipsim.out                      ',
     +       '-file with training image')
      write(lun,13)
 13   format('3   10                            ',
     +       '-debug level, reporting interval')
      write(lun,14)
 14   format('anneal.dbg                        ',
     +       '-file for debug output')
      write(lun,15)
 15   format('anneal.out                        ',
     +       '-file for output simulation')
      write(lun,16)
 16   format('1                                 ',
     +       '-number of realizations')
      write(lun,17)
 17   format(' 25   25  1                       ',
     +       '-nx, ny, nz')
      write(lun,18)
 18   format('69069                             ',
     +       '-random number seed')
      write(lun,19)
 19   format('5   0.000001                      ',
     +       '-maximum iterations, tolerance')
      write(lun,20)
 20   format('4                                 ',
     +       '-number of directions')
      write(lun,21)
 21   format('1  0  0  10                       ',
     +       '-ixl, iyl, izl, nlag')
      write(lun,22)
 22   format('0  1  0  10                       ',
     +       '-ixl, iyl, izl, nlag')
      write(lun,23)
 23   format('1  1  0   5                       ',
     +       '-ixl, iyl, izl, nlag')
      write(lun,24)
 24   format('1 -1  0   5                       ',
     +       '-ixl, iyl, izl, nlag')

      close(lun)
      return
      end



      double precision function acorni(idum)
c-----------------------------------------------------------------------
c
c Fortran implementation of ACORN random number generator of order less
c than or equal to 12 (higher orders can be obtained by increasing the
c parameter value MAXORD).
c
c
c NOTES: 1. The variable idum is a dummy variable. The common block
c           IACO is used to transfer data into the function.
c
c        2. Before the first call to ACORN the common block IACO must
c           be initialised by the user, as follows. The values of
c           variables in the common block must not subsequently be
c           changed by the user.
c
c             KORDEI - order of generator required ( must be =< MAXORD)
c
c             MAXINT - modulus for generator, must be chosen small
c                      enough that 2*MAXINT does not overflow
c
c             ixv(1) - seed for random number generator
c                      require 0 < ixv(1) < MAXINT
c
c             (ixv(I+1),I=1,KORDEI)
c                    - KORDEI initial values for generator
c                      require 0 =< ixv(I+1) < MAXINT
c
c        3. After initialisation, each call to ACORN generates a single
c           random number between 0 and 1.
c
c        4. An example of suitable values for parameters is
c
c             KORDEI   = 10
c             MAXINT   = 2**30
c             ixv(1)   = an odd integer in the (approximate) range 
c                        (0.001 * MAXINT) to (0.999 * MAXINT)
c             ixv(I+1) = 0, I=1,KORDEI
c
c
c
c Author: R.S.Wikramaratna,                           Date: October 1990
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter (KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
      common/iaco/ ixv(MAXOP1)
      do i=1,KORDEI
            ixv(i+1)=(ixv(i+1)+ixv(i))
            if(ixv(i+1).ge.MAXINT) ixv(i+1)=ixv(i+1)-MAXINT
      end do
      acorni=dble(ixv(KORDEI+1))/MAXINT
      return
      end
