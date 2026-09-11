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
c                      Univariate Transformation
c                      *************************
c
c Reads in a reference distribution and a number of other distributions
c and then transforms the values in each of the second distributions
c such that their histograms match that of the reference distribution.
c
c
c
c INPUT/OUTPUT Parameters:
c
c   ivtype      variable type (1=continuous, 0=categorical)
c   distin      file with reference distribution
c   ivr,iwt     columns for variable and weight(0=none)
c   datafl      file with uncorrected distributions
c   ivr,iwt     columns for variable and weight(0=none)
c   tmin,tmax   trimming limits
c   outfl       file for revised distributions
c   nsim        size to transform, number of realizations
c   nx, ny, nz  size of categorical variable realizations to transform
c   wx, wy, wz  window size for breaking ties
c   nxyz        size to of continuous variable data set to transform
c   zmin,zmax   minimum and maximum data values
c   ltail,ltpar lower tail: option, parameter
c   utail,utpar upper tail: option, parameter
c   ldata       honor local data (1=yes, 0=no)
c   localfl     file with estimation variance
c   ikv         column number
c   wtfac       control parameter
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MAXCAT=24,MV=500,EPSLON=1.0e-12,VERSION=3.000)

      character distin*512,datafl*512,outfl*512,localfl*512,str*512
      real      catcdf(MAXCAT),var(MV),ltpar,utpar
      integer   ltail,utail,nx,ny,nz,wx,wy,wz,category(MAXCAT),test
      logical   testfl
c
c Declare dynamic arrays:
c
      real, allocatable :: dcdf(:),dvr(:),indx(:),fuzzcat(:),
     +                     rcdf(:),rvr(:)
c
c For the random number generator:
c
      parameter(KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
      real*8  acorni
      common /iaco/ ixv(MAXOP1)
      data   ixv/MAXOP1*0.0/
      data   lin/1/,lout/2/,lkv/3/
c
c Note VERSION number before anything else:
c
      write(*,9999) VERSION
 9999 format(/' TRANS Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'trans.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'trans.par           ') then
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

      read(lin,*,err=97) ivtype
      write(*,*) ' variable type = ',ivtype

      read(lin,'(a512)',err=97) distin
      call chknam(distin,512)
      write(*,*) ' reference distribution = ',distin(1:40)

      read(lin,*,err=97) ivr,iwt
      write(*,*) ' columns = ',ivr,iwt

      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=97) ivrd,iwtd
      write(*,*) ' columns = ',ivrd,iwtd

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=97) outfl 
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) nsim
      write(*,*) ' number of sets to transform = ',nsim

      read(lin,*,err=97) nx,ny,nz
      write(*,*) ' size of model to transform = ',nx,ny,nz

      read(lin,*,err=97) wx,wy,wz
      write(*,*) ' window size of model to transform = ',wx,wy,wz

      read(lin,*,err=97) nxyz
      write(*,*) ' size of continuous variable data set = ',nxyz

      read(lin,*,err=97) zmin,zmax
      write(*,*) ' data limits = ',zmin,zmax

      read(lin,*,err=97) ltail,ltpar
      write(*,*) ' lower tail option = ',ltail,ltpar

      read(lin,*,err=97) utail,utpar
      write(*,*) ' upper tail option = ',utail,utpar

      read(lin,*,err=97) ldata
      write(*,*) ' account for local data (1=yes) = ',ldata

      if(ldata.eq.1) then

            read(lin,'(a512)',err=97) localfl 
            call chknam(localfl,512)
            write(*,*) ' local file = ',localfl(1:40)

            read(lin,*,err=97) icoll
            write(*,*) ' column for kriging variance = ',icoll

            read(lin,*,err=97) wtfac
            write(*,*) ' scaling factor = ',wtfac
c
c Scale from 0 to 1  --> 0.33 to 3.0
c
            wtfac = 0.33 + wtfac*(3.0-0.33)

            read(lin,*,err=97) ixv(1)
            write(*,*) ' random number seed = ',ixv(1)
            do i=1,10000
                  rn = real(acorni(idum))
            end do

      end if

      close(lin)
c
c Check for error situation:
c
      if(ltail.ne.1.and.ltail.ne.2) then
            write(*,*) 'ERROR invalid lower tail option ',ltail
            write(*,*) '      only allow 1 or 2 - see manual '
            stop
      endif
      if(utail.ne.1.and.utail.ne.2.and.utail.ne.4) then
            write(*,*) 'ERROR invalid upper tail option ',ltail
            write(*,*) '      only allow 1,2 or 4 - see manual '
            stop
      endif
      if(utail.eq.4.and.utpar.lt.1.0) then
            write(*,*) 'ERROR invalid power for hyperbolic tail',utpar
            write(*,*) '      must be greater than 1.0!'
            stop
      endif
c
c Read in the reference distribution:
c
      inquire(file=distin,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: No reference distribution file'
            stop
      endif
      open(lin,file=distin,status='UNKNOWN')
c
c Proceed with reading in distribution:
c
      read(lin,*,err=98)
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxref = 0
 22   read(lin,*,end =44,err=99)(var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 22
      maxref = maxref + 1
      go to 22
 44   continue
c
c Allocate the needed memory:
c
      allocate (rcdf(maxref),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (rvr(maxref),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'()',err=98)
      end do
c
c Read as much data for reference distribution as possible:
c
      ncut = 0
      tcdf = 0
 2    read(lin,*,end=3,err=98) (var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax) go to 2

      vr = var(ivr)
      wt = 1.0
      if(iwt.ge.1) wt = var(iwt)

      if(ivtype.eq.1) then
            ncut = ncut + 1
            rvr(ncut)  = vr
            rcdf(ncut) = wt
      else
            do i=1,ncut
                  if(int(vr+0.5).eq.int(rvr(i))) then
                        icut = i
                        go to 4
                  end if
            end do
            ncut = ncut + 1
            if(ncut.gt.MAXCAT) then
                  write(*,*) 'ERROR: exceeded available storage for'
                  write(*,*) '       categories, available: ',MAXCAT
                  stop
            endif
            rvr(ncut)  = vr
            rcdf(ncut) = 0.0
            icut       = ncut
 4          continue
            rcdf(icut) = rcdf(icut) + wt
      end if
      tcdf = tcdf + wt

c
c Go back for another data?
c
      go to 2
 3    close(lin)
c
c Sort the Reference Distribution and Check for error situation:
c
      call sortem(1,ncut,rvr,1,rcdf,c,d,e,f,g,h)
      if(ncut.le.1.or.tcdf.le.EPSLON) then
            write(*,*) 'ERROR: too few data or too low weight'
            stop
      endif
      if(ivtype.eq.1.and.utail.eq.4.and.rvr(ncut).le.0.0) then
            write(*,*) 'ERROR can not use hyperbolic tail with '
            write(*,*) '      negative values! - see manual '
            stop
      endif
c
c Turn the (possibly weighted) distribution into a cdf that is useful:
c
      tcdf  = 1.0 / tcdf
      if(ivtype.eq.1) then
            oldcp = 0.0
            cp    = 0.0
            do i=1,ncut
                  cp     = cp + rcdf(i) * tcdf
                  rcdf(i) =(cp + oldcp) * 0.5
                  oldcp  = cp
            end do
      else
            do i=1,ncut
                  rcdf(i) = rcdf(i) * tcdf
            end do
      end if
c
c Write Some of the Statistics to the screen:
c
      if(ivtype.eq.1) then
            call locate(rcdf,ncut,1,ncut,0.5,j)
            gmedian = powint(rcdf(j),rcdf(j+1),rvr(j),rvr(j+1),0.5,1.0)
            write(*,900) ncut,gmedian
 900        format(/' There are ',i8,' data in reference dist,',/,
     +              '   median value        = ',f12.5)
      else
            write(*,*)
            do i=1,ncut
                  catcdf(i) = 0.0
                  write(*,901) i,rvr(i),rcdf(i)
 901              format(' Category number ',i2,' code ',f6.0,
     +                   ' proportion ',f6.4)
                  if(i.gt.1) rcdf(i) = rcdf(i) + rcdf(i-1)
            end do
      end if
c
c Get the output and distribution files ready:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR ',datafl,' does not exist!'
            write(*,*) '  you need some distributions to work with'
            stop
      endif
      open(lin,file=datafl,status='OLD')
      open(lout,file=outfl,status='UNKNOWN')
      read(lin,*,err=96)
      read(lin,*,err=96) nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0  
 20   read(lin,*,end =40,err=99)(var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 20
      maxdat = maxdat + 1
      go to 20
 40   continue
c
c Allocate the needed memory:
c
      allocate (dcdf(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (dvr(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (indx(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (fuzzcat(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a40)',err=96) str
      read(lin,*,err=96) nvari
      do i=1,nvari
            read(lin,'(a40)',err=96) str
      end do
      write(lout,110)
 110  format('Trans output',/,'1',/,'transformed variable')
c
c MAIN Loop over all the increments to transform:
c 
      if(ivtype.eq.0) then
            nxyz = nx*ny*nz
            nxy  = nx*ny
      end if
      do isim=1,nsim
c
c Read in the data values:
c
            tcdf = 0.0
            num  = 0
            do i=1,nxyz
                  read(lin,*,end=5,err=98) (var(j),j=1,nvari)
                  num = num + 1
                  dvr(num)  = var(ivrd)
                  indx(num) = real(num)
                  wt        = 0.0
                  if(dvr(num).ge.tmin.and.dvr(num).lt.tmax) then
                        wt = 1.0       
                        if(iwtd.ge.1) wt = var(iwtd)
                        dcdf(num) = wt
                        tcdf      = tcdf + wt
                  endif
c
c Keep track of the proportions if working with a categorical variable:
c
                  if(dvr(num).ge.tmin.and.dvr(num).lt.tmax.and.
     +               ivtype.eq.0) then
                        do j=1,ncut
                           if(int(dvr(num)+0.5).eq.int(rvr(j))) then
                                 icut = j
                                 go to 6
                           end if
                        end do
                        write(*,*) 'Found a code not found in reference'
                        write(*,*) dvr(num)
                        stop
 6                      continue
                        catcdf(icut)  = catcdf(icut) + wt
                  end if
            end do
 5          continue
            if(tcdf.le.EPSLON) then
                  write(*,*) 'ERROR: no data'
                  stop
            endif
            if(ivtype.eq.0.and.num.ne.nxyz) then
                  write(*,*) 'ERROR: you must have ',nxyz
                  write(*,*) '       for a categorical transformation'
                  stop
            endif
c
c For categorical transformation we need a fuzzy category which is the
c local average category - this is for breaking ties in the
c transformation procedure.
c
            if(ivtype.eq.0) then
                  do i=1,nxyz
                        fuzzcat(i) = 0.0
                        nloc       = 0
                        iz = int((i-1)/nxy) + 1
                        iy = int((i-(iz-1)*nxy-1)/nx) + 1
                        ix = i- (iz-1)*nxy - (iy-1)*nx
                        do iix=-wx,wx
                        do iiy=-wy,wy
                        do iiz=-wz,wz
                              jx = ix + iix
                              jy = iy + iiy
                              jz = iz + iiz
                              if(jx.ge.1.and.jx.le.nx.and.
     +                           jy.ge.1.and.jy.le.ny.and.
     +                           jz.ge.1.and.jz.le.nz) then
                                 j = jx + (jy-1)*nx + (jz-1)*nxy
                                 if(dvr(j).ge.tmin.and.
     +                              dvr(j).lt.tmax) then
                                       nloc = nloc + 1
                                       fuzzcat(i) = fuzzcat(i) + dvr(j)
                                 end if
                              end if
                        end do
                        end do
                        end do
                        if(nloc.gt.0) then
                              fuzzcat(i) = fuzzcat(i) / real(nloc)
                        else
                              fuzzcat(i) = 2*tmax
                        end if
                  end do
            end if
c
c Turn the (possibly weighted) data distribution into a useful cdf:
c
            if(ivtype.eq.1) then
                  call sortem(1,num,dvr,2,dcdf,indx,d,e,f,g,h)
            else
                  call sortem(1,num,fuzzcat,3,dcdf,dvr,indx,e,f,g,h)
            end if
            oldcp = 0.0
            cp    = 0.0
            tcdf  = 1.0 / tcdf
            do i=1,num
                  if(dvr(i).ge.tmin.and.dvr(i).lt.tmax) then
                        cp     = cp + dcdf(i) * tcdf
                        dcdf(i) =(cp + oldcp) * 0.5
                        oldcp  = cp
                  endif
            end do
c
c Now, get the right order back:
c
            if(ivtype.eq.1) then
                  call sortem(1,num,indx,2,dcdf,dvr,d,e,f,g,h)
            else
                  call sortem(1,num,indx,3,dcdf,dvr,fuzzcat,e,f,g,h)
            end if
c
c Read in the kriging variance to array "indx" if we have to honor
c local data:
c
            if(ldata.eq.1) then
                  open(lkv,file=localfl,err=95,status='OLD')
                  read(lkv,'()',err=95)
                  read(lkv,*,   err=95) nvarik
                  do i=1,nvarik
                        read(lkv,'()',err=95)
                  end do
                  evmax = -1.0e21
                  do i=1,num
                        read(lkv,*,err=95) (var(j),j=1,nvarik)
                        indx(i) = var(icoll)
                        if(indx(i).ge.0.0.and.indx(i).lt.tmax) then
                              indx(i) = sqrt(max(indx(i),0.0))
                              if(indx(i).gt.evmax) evmax = indx(i)
                        end if
                  end do
                  close(lkv)
            end if
c
c Go through all the data back transforming them to the reference CDF:
c
            do i=1,num
                  if(dvr(i).ge.tmin.and.dvr(i).lt.tmax) then
                        if(ivtype.eq.1) then
                              zval = getz(dcdf(i),ncut,rvr,rcdf,zmin,
     +                                    zmax,ltail,ltpar,utail,utpar)
                        else
                              zval = rvr(1)
                              do j=2,ncut
                                    if(dcdf(i).gt.rcdf(j-1))
     +                              zval = rvr(j)
                              end do
                        end if
c
c Now, do we have to honor local data?
c
                        if(ldata.eq.1) then
                              if(indx(i).ge.0.and.indx(i).lt.tmax) then
                                    wt = (indx(i)/evmax)**wtfac
                                    if(ivtype.eq.1) then
                                          zval = dvr(i)+wt*(zval-dvr(i))
                                    else
                                          rn = real(acorni(idum))
                                          if(rn.gt.wt) zval = dvr(i)
                                    end if
                              end if
                        end if
                  else
                        zval = -999.0
                  endif
                  if(ivtype.eq.1) then
                        call numtext(zval,str(1:12))
                        write(lout,'(a12)') str(1:12)
                  else
                        write(lout,'(f4.0)') zval
                  end if
            end do
c
c END Main loop:
c
      end do
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' TRANS Version: ',f5.3, ' Finished'/)
      stop
 95   stop 'ERROR in kriging variance file!'
 96   stop 'ERROR in distribution file!'
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in global data file!'
 99   stop 'ERROR in data file!'
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
      open(lun,file='trans.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for TRANS',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('1                         ',
     +       '-1=continuous, 0=categorical')
      write(lun,12)
 12   format('../data/true.dat          ',
     +       '-file with reference distribution')
      write(lun,13)
 13   format('1   0                     ',
     +       '-   columns for variable and weight(0=none)')
      write(lun,14)
 14   format('../data/cluster.dat       ',
     +       '-file with original distributions')
      write(lun,15)
 15   format('3   0                     ',
     +       '-   columns for variable and weight(0=none)')
      write(lun,16)
 16   format('-1.0e21  1.0e21           ',
     +       '-trimming limits')
      write(lun,17)
 17   format('trans.out                 ',
     +       '-file for transformed distributions')
      write(lun,18)
 18   format('1                         ',
     +       '-number of realizations or "sets" to trans')
      write(lun,19)
 19   format('50  50  1                 ',
     +       '-categorical: nx, ny, nz: size of 3-D model')
      write(lun,20)
 20   format(' 2   2  0                 ',
     +       '-   wx, wy, wz: window size for tie-breaking')
      write(lun,21)
 21   format('1000                      ',
     +       '-continuous: number to transform per "set"')
      write(lun,22)
 22   format('0.0   75.0                ',
     +       '-   minimum and maximum values')
      write(lun,23)
 23   format('1      1.0                ',
     +       '-   lower tail: option, parameter')
      write(lun,24)
 24   format('1     75.0                ',
     +       '-   upper tail: option, parameter')
      write(lun,25)
 25   format('0                         ',
     +       '-honor local data? (1=yes, 0=no)')
      write(lun,26)
 26   format('kt3d.out                  ',
     +       '-   file with estimation variance')
      write(lun,27)
 27   format('2                         ',
     +       '-   column number')
      write(lun,28)
 28   format('0.5                       ',
     +       '-   control parameter ( 0.33 < w < 3.0 )')
      write(lun,29)
 29   format('69069                     ',
     +       '-   random number seed (conditioning cat.)')

      close(lun)
      return
      end
