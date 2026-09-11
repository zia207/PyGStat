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
      
      real,allocatable         :: vr(:)
      real*8,allocatable       :: sills(:),gam(:),hm(:),tm(:),hv(:),
     +                            tv(:),np(:)
      integer,allocatable      :: ixd(:),iyd(:),izd(:),ivtail(:),
     +                            ivhead(:),ivtype(:)
      character*12,allocatable :: names(:)

      real EPSLON,VERSION,xsiz,ysiz,zsiz,tmin,tmax
      integer   nlag,nx,ny,nz,nxy,nxyz,ndir,isill,nvarg,test
      character outfl*512

      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c                Variogram of Data on a Regular Grid
c                ***********************************
c
c This is a template driver program for GSLIB's "gam" subroutine.  The
c input data is ordered rowwise (x cycles fastest, then y, then z) in a
c GEOEAS format file.  The User's Guide contains more details.
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example gam.par)
c
c
c The output file will contain each directional variogram ordered by
c direction and then variogram (the directions cycle fastest then the
c variogram number).  For each variogram there will be a one line
c description and then "nlag" lines with the following:
c
c        a) lag number (increasing from 1 to nlag)
c        b) separation distance
c        c) the "variogram" value
c        d) the number of pairs for the lag
c        e) the mean of the data contributing to the tail
c        f) the mean of the data contributing to the head
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
c
c Read the Parameter File:
c
      use geostat
      EPSLON  = 1.0e-20
      VERSION = 3.000
      call readparm
c
c Call gam to compute the required variograms:
c
      call gamma
c
c Write Results:
c
      call writeout
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' GAM Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
 
      subroutine readparm
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters and data are read in from their files. Some quick
c error checking is performed and the statistics of all the variables
c being considered are written to standard output.
c
c
c
c-----------------------------------------------------------------------
      use msflib
      use geostat
      parameter(MV=500)
      real      var(MV),cut(500)
      real*8    avg(MV),ssq(MV)
      integer   ivar(MV),num(MV),ivc(500),indflag(500)
      character datafl*512,str*512
      logical   testfl
      data      lin/1/,ncut/0/
c
c Declare dynamic arrays:
c
      real,allocatable :: vrmin(:),vrmax(:)
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' GAM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'gam.par             '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'gam.par             ') then
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

      read(lin,*,err=98) nvar
      write(*,*) ' number of variables = ',nvar
      backspace lin

      read(lin,*,err=98) j,(ivar(i),i=1,nvar)
      write(*,*) ' columns = ',(ivar(i),i=1,nvar)

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) isim
      write(*,*) ' grid number = ',isim

      read(lin,*,err=98) nx,xmn,xsiz
      write(*,*) ' nx,xmn,xsiz = ',nx,xmn,xsiz

      read(lin,*,err=98) ny,ymn,ysiz
      write(*,*) ' ny,ymn,ysiz = ',ny,ymn,ysiz

      read(lin,*,err=98) nz,zmn,zsiz
      write(*,*) ' nz,zmn,zsiz = ',nz,zmn,zsiz

      nxy  = nx * ny
      nxyz = nx * ny * nz
      if(nx.lt.1)         stop 'nx must be at least 1: check parameters'
      if(ny.lt.1)         stop 'ny must be at least 1: check parameters'
      if(nz.lt.1)         stop 'nz must be at least 1: check parameters'
     
      read(lin,*,err=98) ndir,nlag
      write(*,*) ' ndir,nlag = ',ndir,nlag
      if(ndir.lt.1)       stop 'ndir is too small: check parameters'
      if(nlag.lt.1)       stop 'nlag is too small: check parameters'
     
      allocate (ixd(ndir),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if  
      
      allocate (iyd(ndir),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if  
     
      allocate (izd(ndir),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if  
      
      do i=1,ndir
            read(lin,*,err=98) ixd(i),iyd(i),izd(i)
            write(*,*) ' direction = ',ixd(i),iyd(i),izd(i)
      end do
      read(lin,*,err=98) isill
      write(*,*) ' flag to standardize sills = ',isill

      read(lin,*,err=98) nvarg
      write(*,*) ' number of variograms = ',nvarg
      if(nvarg.lt.1)      stop 'nvarg is too small: check parameters'
c
c Allocate the needed memory:
c
      allocate (names(nvar+nvarg),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      mxdlv = ndir*nlag*nvarg
      allocate (gam(mxdlv),stat = test) 
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (hm(mxdlv),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (tm(mxdlv),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (hv(mxdlv),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (tv(mxdlv),stat = test)  
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (np(mxdlv),stat = test)  
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (ivtail(nvarg+2),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (ivhead(nvarg+2),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
      allocate (ivtype(nvarg+2),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if      
c
      ncut = 0
      do i=1,nvarg
            read(lin,*,err=98) ivtail(i),ivhead(i),ivtype(i)
            write(*,*) ' tail,head,type = ',
     +                   ivtail(i),ivhead(i),ivtype(i)
            if(ivtype(i).eq.9.or.ivtype(i).eq.10) then
                   ncut = ncut + 1
                   if(tmin.gt.0.0)stop'tmin interferes with indicators!'
                   if(tmax.le.1.0)stop'tmax interferes with indicators!'
                   backspace lin
                   read(lin,*,err=98) ii,jj,kk,cut(ncut)
                   if(ivtype(i).eq.9)  indflag(ncut) = 1
                   if(ivtype(i).eq.10) indflag(ncut) = 0
                   ivc(ncut) = ivtail(i)
                   ivtail(i) = nvar + ncut
                   ivhead(i) = nvar + ncut
                   write(names(nvar+ncut),140) ncut
 140               format('Indicator ',i2)
                   write(*,*) ' indicator threshold:  ',cut(ncut)
            endif
      end do
      write(*,*)
      close(lin)
c
c Determine the maximum size needed and allocate the needed memory:
c
      MAXVAR = nvar + ncut
      allocate (vr(nxyz*MAXVAR),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      allocate (sills(MAXVAR),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      allocate (vrmin(MAXVAR),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (vrmax(MAXVAR),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
c Check to make sure the data file exists, then either read in the
c data or write an error message and stop:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR data file ',datafl,' does not exist!'
            stop
      endif
c
c The data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99)       nvari
      do i=1,nvari
            read(lin,'(a40)',err=99) str
            do iv=1,nvar
                  j=ivar(iv)
                  if(i.eq.j) names(iv) = str(1:12)
            end do
            num(i) = 0
            avg(i) = 0.0
            ssq(i) = 0.0
      end do
c
c Read the regular grid information row wise (x cycles fastest):
c
      do is=1,isim
            do iz=1,nz
            do iy=1,ny
            do ix=1,nx
                  if(is.ne.isim) then
                        read(lin,'()',err=99)
                  else
                        read(lin,*,err=99) (var(i),i=1,nvari)
                        do iv=1,nvar
                              i=ivar(iv)
                              index=ix+(iy-1)*nx+(iz-1)*nxy+(iv-1)*nxyz
                              vr(index) = var(i)
                              if(var(i).ge.tmin.and.var(i).lt.tmax) then
                                    num(iv) = num(iv) + 1
                                    avg(iv) = avg(iv) + dble(var(i))
                                    ssq(iv) = ssq(iv)
     +                                      + dble(var(i)*var(i))
                              end if
                        end do
                  end if
            end do
            end do
            end do
      end do
      close(lin)
c
c Compute the averages and variances as an error check for the user:
c
      do iv=1,nvar
            sills(iv) = -999.
            if(num(iv).gt.0) then
                  avg(iv) = avg(iv) / dble(num(iv))
                  ssq(iv) =(ssq(iv) / dble(num(iv))) - avg(iv) * avg(iv)
                  sills(iv) = ssq(iv)
                  write(*,*) 'Variable number ',iv
                  write(*,*) '  Number   = ',num(iv)
                  write(*,*) '  Average  = ',real(avg(iv))
                  write(*,*) '  Variance = ',real(ssq(iv))
            endif
      end do
c
c Construct Indicator Variables if necessary:
c
      do ic=1,ncut
            iv   = ivc(ic)
            jv   = nvar + ic
            ptot = 0.0
            p1   = 0.0
            do ix=1,nx
            do iy=1,ny
            do iz=1,nz
                  index = ix+(iy-1)*nx+(iz-1)*nxy+(iv-1)*nxyz
                  jndex = ix+(iy-1)*nx+(iz-1)*nxy+(jv-1)*nxyz
                  if(vr(index).lt.tmin.or.vr(index).ge.tmax) then
                        vr(jndex) = tmin - EPSLON
                  else
                        if(indflag(ic).eq.1) then
                              if(vr(index).lt.cut(ic)) then
                                    vr(jndex) = 0.0
                              else
                                    vr(jndex) = 1.0
                              end if
                              p1   = p1   + vr(index)
                              ptot = ptot + 1.0
                        else
                              vr(jndex) = 0.0
                              if(int(vr(index)+0.5).eq.int(cut(ic)+0.5))
     +                        vr(jndex) = 1.0
                              p1   = p1   + vr(jndex)
                              ptot = ptot + 1.0
                        end if
                  end if
            end do
            end do
            end do
            p1        = p1 / max(ptot,1.0)
            sills(jv) = dble (p1*(1.0-p1))
      end do
c
c Establish minimums and maximums:
c
      do i=1,MAXVAR
            vrmin(i) =  1.0e21
            vrmax(i) = -1.0e21
      end do
      do ix=1,nx
       do iy=1,ny
        do iz=1,nz
         do iv=1,nvar+ncut
          index = ix+(iy-1)*nx+(iz-1)*nxy+(iv-1)*nxyz
          if(vr(index).ge.tmin.and.vr(index).lt.tmax) then
            if(vr(index).lt.vrmin(iv)) vrmin(iv) = vr(index)
            if(vr(index).gt.vrmax(iv)) vrmax(iv) = vr(index)
          end if
         end do 
        end do 
       end do 
      end do 
c
c Check on the variogams that were requested:
c
      call check(vrmin,vrmax)
c
c Return:
c
      return
c
c Error in an Input File Somewhere:
c
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine gamma
c-----------------------------------------------------------------------
c
c                Variogram of Data on a Regular Grid
c                ***********************************
c
c This subroutine computes any of eight different measures of spatial
c continuity for regular spaced 3-D data.  Missing values are allowed
c and the grid need not be cubic.
c
c
c
c INPUT VARIABLES:
c
c   nlag             Maximum number of lags to be calculated
c   nx               Number of units in x (number of columns)
c   ny               Number of units in y (number of lines)
c   nz               Number of units in z (number of levels)
c   ndir             Number of directions to consider
c   ixd(ndir)        X (column) indicator of direction - number of grid
c                      columns that must be shifted to move from a node
c                      on the grid to the next nearest node on the grid
c                      which lies on the directional vector
c   iyd(ndir)        Y (line) indicator of direction - similar to ixd,
c                      number of grid lines that must be shifted to
c                      nearest node which lies on the directional vector
c   izd(ndir)        Z (level) indicator of direction - similar to ixd,
c                      number of grid levels that must be shifted to
c                      nearest node of directional vector
c   nv               The number of variables
c   vr(nx*ny*nz*nv)  Array of data
c   tmin,tmax        Trimming limits
c   isill            1=attempt to standardize, 0=do not
c   sills            the sills (variances) to standardize with
c   nvarg            Number of variograms to compute
c   ivtail(nvarg)    Variable for the tail of the variogram
c   ivhead(nvarg)    Variable for the head of the variogram
c   ivtype(nvarg)    Type of variogram to compute:
c                      1. semivariogram
c                      2. cross-semivariogram
c                      3. covariance
c                      4. correlogram
c                      5. general relative semivariogram
c                      6. pairwise relative semivariogram
c                      7. semivariogram of logarithms
c                      8. madogram
c                      9. indicator semivariogram: an indicator variable
c                         is constructed in the main program.
c
c OUTPUT VARIABLES:  The following arrays are ordered by direction,
c                    then variogram, and finally lag, i.e.,
c                      iloc = (id-1)*nvarg*nlag+(iv-1)*nlag+il
c
c   np()             Number of pairs
c   gam()            Semivariogram, covariance, correlogram,... value
c   hm()             Mean of the tail data
c   tm()             Mean of the head data
c   hv()             Variance of the tail data
c   tv()             Variance of the head data
c
c
c
c Original:  A.G. Journel                                           1978
c Revisions: B.E. Buxton                                       Apr. 1982
c-----------------------------------------------------------------------
      use geostat
c
c Initialize the summation arrays for each direction, variogram, and lag
c
      nxyz = nx*ny*nz
      nsiz = ndir*nvarg*nlag
      do i=1,nsiz
            np(i)  = 0.
            gam(i) = 0.0
            hm(i)  = 0.0
            tm(i)  = 0.0
            hv(i)  = 0.0
            tv(i)  = 0.0
      end do
c
c First fix the location of a seed point on the grid (ix,iy,iz):
c
      do ix=1,nx
      do iy=1,ny
      do iz=1,nz
c
c For the fixed seed point, loop through all directions:
c
            do id=1,ndir
              ixinc = ixd(id)
              iyinc = iyd(id)
              izinc = izd(id)
              ix1   = ix
              iy1   = iy
              iz1   = iz
c
c For this direction, loop over all the lags:
c
              do il=1,nlag
c
c Check to be sure that the point being considered is still in the
c grid - if not, then finished with this direction:
c
                ix1 = ix1 + ixinc
                if(ix1.lt.1.or.ix1.gt.nx) go to 3
                iy1 = iy1 + iyinc
                if(iy1.lt.1.or.iy1.gt.ny) go to 3
                iz1 = iz1 + izinc
                if(iz1.lt.1.or.iz1.gt.nz) go to 3
c
c For this direction and lag, loop over all variograms:
c
                do iv=1,nvarg
                  it = ivtype(iv)
c
c Get the head value, skip this value if missing:
c
                  i     = ivhead(iv)
                  index = ix+(iy-1)*nx+(iz-1)*nxy+(i-1)*nxyz
                  vrt   = vr(index)
                  if(vrt.lt.tmin.or.vrt.ge.tmax) go to 5
c
c Get the tail value, skip this value if missing:
c
                  i     = ivtail(iv)
                  index = ix1+(iy1-1)*nx+(iz1-1)*nxy+(i-1)*nxyz
                  vrh   = vr(index)
                  if(vrh.lt.tmin.or.vrh.ge.tmax) go to 5
c
c Need increment for the cross semivariogram only:
c
                  if(it.eq.2) then
                        i     = ivtail(iv)
                        index = ix+(iy-1)*nx+(iz-1)*nxy+(i-1)*nxyz
                        vrhpr = vr(index)
                        if(vrhpr.lt.tmin.or.vrhpr.ge.tmax) go to 5
                        i     = ivhead(iv)
                        index = ix1+(iy1-1)*nx+(iz1-1)*nxy+(i-1)*nxyz
                        vrtpr = vr(index)
                        if(vrtpr.lt.tmin.or.vrtpr.ge.tmax) go to 5
                  endif
c
c We have an acceptable pair, therefore accumulate all the statistics
c that are required for the variogram:
c
                  i      = (id-1)*nvarg*nlag+(iv-1)*nlag+il
                  np(i)  = np(i) + 1.
                  tm(i)  = tm(i) + dble(vrt)
                  hm(i)  = hm(i) + dble(vrh)
c
c Choose the correct variogram type and keep relevant sums:
c
                  if(it.eq.1.or.it.ge.9) then
                    gam(i) = gam(i) + dble((vrh-vrt)*(vrh-vrt))
                  else if(it.eq.2) then
                    gam(i) = gam(i) + dble((vrhpr-vrh)*(vrt-vrtpr))
                  else if(abs(it).eq.3) then
                    gam(i) = gam(i) +  dble(vrh*vrt)
                  else if(it.eq.4) then
                    gam(i) = gam(i) +  dble(vrh*vrt)
                    hv(i)  = hv(i)  +  dble(vrh*vrh)
                    tv(i)  = tv(i)  +  dble(vrt*vrt)
                  else if(it.eq.5) then
                    gam(i) = gam(i) + dble((vrh-vrt)*(vrh-vrt))
                  else if(it.eq.6) then
                    if((vrt+vrh).lt.EPSLON) then
                        np(i)  = np(i) - 1.
                        tm(i)  = tm(i) - dble(vrt)
                        hm(i)  = hm(i) - dble(vrh)
                    else
                        tempvar= 2.0*(vrt-vrh)/(vrt+vrh)
                        gam(i) = gam(i) + dble(tempvar*tempvar)
                    endif
                  else if(it.eq.7) then
                    if(vrt.lt.EPSLON.or.vrh.lt.EPSLON) then
                        np(i)  = np(i) - 1.
                        tm(i)  = tm(i) - dble(vrt)
                        hm(i)  = hm(i) - dble(vrh)
                    else
                        tempvar= alog(vrt)-alog(vrh)
                        gam(i) = gam(i) + dble(tempvar*tempvar)
                    endif
                  else if(it.eq.8) then
                    gam(i) = gam(i) + dble(abs(vrt-vrh))
                  endif
 5              continue
                end do
 4            continue
              end do
 3          continue
            end do
      end do
      end do
      end do
c
c Get average values for gam, hm, tm, hv, and tv, then compute
c the correct "variogram" measure:
c
      do id=1,ndir
      do iv=1,nvarg
      do il=1,nlag
            i = (id-1)*nvarg*nlag+(iv-1)*nlag+il
            if(np(i).eq.0.) go to 6
            rnum   = np(i)
            gam(i) = gam(i) / dble(rnum)
            hm(i)  = hm(i)  / dble(rnum)
            tm(i)  = tm(i)  / dble(rnum)
            hv(i)  = hv(i)  / dble(rnum)
            tv(i)  = tv(i)  / dble(rnum)
            it     = ivtype(iv)
c
c Attempt to standardize:
c

            if(isill.eq.1) then
                  if(ivtail(iv).eq.ivhead(iv)) then
                        iii = ivtail(iv)
                        if((it.eq.1.or.it.ge.9).and.sills(iii).gt.0.0)
     +                    gam(i) = gam(i) / sills(iii)
                  end if
            end if
c
c 1. report the semivariogram rather than variogram
c 2. report the cross-semivariogram rather than variogram
c 3. the covariance requires "centering"
c 4. the correlogram requires centering and normalizing
c 5. general relative requires division by lag mean
c 6. report the semi(pairwise relative variogram)
c 7. report the semi(log variogram)
c 8. report the semi(madogram)
c
            if(it.eq.1.or.it.eq.2) then
                  gam(i) = 0.5 * gam(i)
            else if(abs(it).eq.3) then
                  gam(i) = gam(i) - hm(i)*tm(i)
                  if(it.lt.0) then
                        if(sills(ivtail(iv)).lt.0.0.or.
     +                     sills(ivhead(iv)).lt.0.0) then
                              gam(i) = -999.0
                        else
                              variance = ( sqrt(sills(ivtail(iv)))
     +                                 *   sqrt(sills(ivhead(iv))) )
                              gam(i) = variance - gam(i)
                        end if
                  end if
            else if(it.eq.4) then
                  hv(i)  = hv(i)-hm(i)*hm(i)
                  if(hv(i).le.0.0) hv(i) = 0.0
                  hv(i)  = sqrt(hv(i))
                  tv(i)  = tv(i)-tm(i)*tm(i)
                  if(tv(i).le.0.0) tv(i) = 0.0
                  tv(i)  = sqrt(tv(i))
                  if((hv(i)*tv(i)).lt.EPSLON) then
                        gam(i) = 0.0
                  else
                        gam(i) =(gam(i)-hm(i)*tm(i))/(hv(i)*tv(i))
                  endif
c
c Square "hv" and "tv" so that we return the variance:
c
                  hv(i)  = hv(i)*hv(i)
                  tv(i)  = tv(i)*tv(i)
            else if(it.eq.5) then
                  htave  = 0.5*(hm(i)+tm(i))
                  htave  = htave   *   htave
                  if(htave.lt.EPSLON) then
                        gam(i) = 0.0
                  else
                        gam(i) = gam(i)/dble(htave)
                  endif
            else if(it.ge.6) then
                  gam(i) = 0.5 * gam(i)
            endif
 6    continue
      end do
      end do
      end do
      return
      end
 
 
 
      subroutine writeout
c-----------------------------------------------------------------------
c
c                  Write Out the Results of GAM
c                  ****************************
c
c An output file will be written which contains each directional
c variogram ordered by direction and then variogram (the directions
c cycle fastest then the variogram number).  For each variogram there
c will be a one line description and then "nlag" lines with:
c
c        a) lag number (increasing from 1 to nlag)
c        b) separation distance
c        c) the "variogram" value
c        d) the number of pairs for the lag
c        e) the mean of the data contributing to the tail
c        f) the mean of the data contributing to the head
c        g) IF the correlogram - variance of tail values
c        h) IF the correlogram - variance of head values
c
c
c-----------------------------------------------------------------------
      use geostat

      character title*80
      data      lout/1/
c
c Loop over all the variograms that have been computed:
c
      open(lout,file=outfl,status='UNKNOWN')
c
      do iv=1,nvarg
c
c Construct a title that reflects the variogram type and the variables
c that were used to calculate the variogram:
c
      it = abs(ivtype(iv))
      if(it.eq. 1) title(1:24) = 'Semivariogram           '
      if(it.eq. 2) title(1:24) = 'Cross Semivariogram     '
      if(it.eq. 3) title(1:24) = 'Covariance              '
      if(it.eq. 4) title(1:24) = 'Correlogram             '
      if(it.eq. 5) title(1:24) = 'General Relative        '
      if(it.eq. 6) title(1:24) = 'Pairwise Relative       '
      if(it.eq. 7) title(1:24) = 'Variogram of Logarithms '
      if(it.eq. 8) title(1:24) = 'Semimadogram            '
      if(it.eq. 9) title(1:24) = 'Indicator 1/2 Variogram '
      if(it.eq.10) title(1:24) = 'Indicator 1/2 Variogram '
      write(title(25:62),100) names(ivtail(iv)),names(ivhead(iv))
 100  format('tail:',a12,' head:',a12)
c
c Loop over all the directions (note the direction in the title):
c
      do id=1,ndir
            write(title(62:74),101) id
 101        format('direction ',i2)
            write(lout,'(a74)') title(1:74)
c
c Compute the unit lag distance along the directional vector:
c
            dis = sqrt( max(((ixd(id)*xsiz)**2 + (iyd(id)*ysiz)**2 +
     +                  (izd(id)*zsiz)**2),0.0) )
c
c Write out all the lags:
c
            do il=1,nlag
                  i = (id-1)*nvarg*nlag+(iv-1)*nlag+il
                  disl = real(il)*dis
                  nump = int(np(i))
                  if(it.eq.4) then
                        write(lout,102) il,disl,gam(i),nump,
     +                                  hm(i),tm(i),hv(i),tv(i)
                  else
                        write(lout,102) il,disl,gam(i),nump,
     +                                  hm(i),tm(i)
                  endif
 102              format(1x,i3,1x,f12.3,1x,f12.5,1x,i8,4(1x,f14.5))
            end do
      end do
      end do
      close(lout)
      return
      end
 
 
 
      subroutine check(vrmin,vrmax)
c-----------------------------------------------------------------------
c
c                Error Check and Note Variogram Types
c                ************************************
c
c Go through each variogram type and note the type to the screen and
c report any possible errors.
c
c
c
c
c
c-----------------------------------------------------------------------
      use       geostat
      real      vrmin(*),vrmax(*)
      character title*80
c
c Loop over all the variograms to be computed:
c
      write(*,*)
      do iv=1,nvarg
c
c Note the variogram type and the variables being used:
c
c
      it = abs(ivtype(iv))
      if(it.eq. 1) title(1:24) = 'Semivariogram          :'
      if(it.eq. 2) title(1:24) = 'Cross Semivariogram    :'
      if(it.eq. 3) title(1:24) = 'Covariance             :'
      if(it.eq. 4) title(1:24) = 'Correlogram            :'
      if(it.eq. 5) title(1:24) = 'General Relative       :'
      if(it.eq. 6) title(1:24) = 'Pairwise Relative      :'
      if(it.eq. 7) title(1:24) = 'Variogram of Logarithms:'
      if(it.eq. 8) title(1:24) = 'Semimadogram           :'
      if(it.eq. 9) title(1:24) = 'Indicator 1/2 Variogram:'
      if(it.eq.10) title(1:24) = 'Indicator 1/2 Variogram:'
      write(title(25:64),100) names(ivtail(iv)),names(ivhead(iv))
 100  format('  tail=',a12,' head=',a12)
      write(*,101) iv,title(1:64)
 101  format(' Variogram ',i2,1x,a64)
c
c Check for possible errors or inconsistencies:
c
      if(it.eq.2) then
            if(ivtail(iv).eq.ivhead(iv)) write(*,201)
 201        format('  WARNING: cross variogram with the same variable!')
      else if(it.eq.5) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,501)
            if(vrmin(ivtail(iv)).lt.0.0.and.vrmax(ivtail(iv)).gt.0.0)
     +            write(*,502)
            if(vrmin(ivhead(iv)).lt.0.0.and.vrmax(ivhead(iv)).gt.0.0)
     +            write(*,502)
 501        format('  WARNING: cross general relative variogram are',
     +             ' difficult to interpret!')
 502        format('  WARNING: there are both positive and negative',
     +             ' values - lag mean could be zero!')
      else if(it.eq.6) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,601)
            if(vrmin(ivtail(iv)).lt.0.0.and.vrmax(ivtail(iv)).gt.0.0)
     +            write(*,602)
            if(vrmin(ivhead(iv)).lt.0.0.and.vrmax(ivhead(iv)).gt.0.0)
     +            write(*,602)
 601        format('  WARNING: cross pairwise relative variogram are',
     +             ' difficult to interpret!')
 602        format('  WARNING: there are both positive and negative',
     +             ' values - pair means could be zero!')
      else if(it.eq.7) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,701)
            if(vrmin(ivtail(iv)).lt.0.0.or.vrmin(ivhead(iv)).lt.0.0)
     +      write(*,702)
 701        format('  WARNING: cross logarithmic variograms may be',
     +             ' difficult to interpret!')
 702        format('  WARNING: there are zero or negative',
     +             ' values - logarithm undefined!')
      else if(it.eq.8) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,901)
 901        format('  WARNING: cross madograms may be difficult to',
     +             ' interpret!')
      endif
c
c Loop over all variograms:
c
      end do
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
      open(lun,file='gam.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for GAM',/,
     +       '                  ******************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/true.dat      ',
     +       '-file with data')
      write(lun,12)
 12   format('2   1   2             ',
     +       '-   number of variables, column numbers')
      write(lun,13)
 13   format('-1.0e21     1.0e21    ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('gam.out               ',
     +       '-file for variogram output')
      write(lun,15)
 15   format('1                     ',
     +       '-grid or realization number')
      write(lun,16)
 16   format('50   0.5   1.0        ',
     +       '-nx, xmn, xsiz')
      write(lun,17)
 17   format('50   0.5   1.0        ',
     +       '-ny, ymn, ysiz')
      write(lun,18)
 18   format(' 1   0.5   1.0        ',
     +       '-nz, zmn, zsiz')
      write(lun,19)
 19   format('2  10                 ',
     +       '-number of directions, number of lags')
      write(lun,20)
 20   format(' 1  0  0              ',
     +       '-ixd(1),iyd(1),izd(1)')
      write(lun,21)
 21   format(' 0  1  0              ',
     +       '-ixd(2),iyd(2),izd(2)')
      write(lun,22)
 22   format('1                     ',
     +       '-standardize sill? (0=no, 1=yes)')
      write(lun,23)
 23   format('5                     ',
     +       '-number of variograms')
      write(lun,24)
 24   format('1   1   1             ',
     +       '-tail variable, head variable, variogram type')
      write(lun,25)
 25   format('1   1   3             ',
     +       '-tail variable, head variable, variogram type')
      write(lun,26)
 26   format('2   2   1             ',
     +       '-tail variable, head variable, variogram type')
      write(lun,27)
 27   format('2   2   3             ',
     +       '-tail variable, head variable, variogram type')
      write(lun,28)
 28   format('1   1   9  2.5        ',
     +       '-tail variable, head variable, variogram type')
      write(lun,40)
 40   format(//,'type 1 = traditional semivariogram',/,
     +          '     2 = traditional cross semivariogram',/,
     +          '     3 = covariance',/,
     +          '     4 = correlogram',/,
     +          '     5 = general relative semivariogram',/,
     +          '     6 = pairwise relative semivariogram',/,
     +          '     7 = semivariogram of logarithms',/,
     +          '     8 = semimadogram',/,
     +          '     9 = indicator semivariogram - continuous',/,
     +          '     10= indicator semivariogram - categorical')

      close(lun)
      return
      end
