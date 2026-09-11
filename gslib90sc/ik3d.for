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
      
      integer,allocatable :: nisb(:),it(:),nst(:),nviol(:),
     +        ixsbtosr(:),iysbtosr(:),izsbtosr(:)
      real,allocatable    :: x(:),y(:),z(:),vr(:,:),tmp(:),xa(:),
     +        ya(:),za(:),vra(:),close(:),actloc(:),gcdf(:),sb(:),
     +        ccdf(:),ccdfo(:),aviol(:),xviol(:),thres(:),c0(:),cc(:),
     +        aa(:),ang1(:),ang2(:),ang3(:),anis1(:),anis2(:),dh(:),
     +        sdis(:)
      real*8,allocatable  :: r(:),s(:),a(:),rotmat(:,:,:)
      
      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c              Indicator Kriging of a 3-D Rectangular Grid
c              *******************************************
c
c This is a template driver program for GSLIB's "ik3d" subroutine. The
c input data must be entered with coordinates in a GEOEAS format file.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use geostat
      include  'ik3d.inc'
c
c Read the Parameters and Data (the output files are also opened):
c
      call readparm(MAXCUT,MAXROT,MAXSBX,MAXSBY,MAXSBZ)
c
c Call ik3d to krige the grid:
c
      call ik3d(MAXCUT,MAXROT,MAXSBX,MAXSBY,MAXSBZ)
c
c Finished:
c
      close(ldbg)
      close(lout)
      write(*,9998) VERSION
 9998 format(/' IK3D Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
 
      subroutine readparm(MAXCUT,MAXROT,MAXSBX,MAXSBY,MAXSBZ)
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
      include  'ik3d.inc'
      parameter(MV=100)
      integer   ivrs(MAXCUT)
      real      var(MV)
      character datafl*512,softfl*512,outfl*512,dbgfl*512,jackfl*512,
     +          str*512
      logical   testfl
c
c Input/Output Units:
c
      lin   = 1
      ldbg  = 3
      lout  = 4
      ljack = 9
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' IK3D Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'ik3d.par            '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'ik3d.par            ') then
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

      read(lin,*,err=98) ivtype
      write(*,*) ' variable type (1=continuous, 0=categorical)= ',ivtype

      read(lin,*,err=98) koption
      write(*,*) ' kriging option (1=cross v, 0=grid)= ',koption

      read(lin,'(a512)',err=98) jackfl
      call chknam(jackfl,512)
      write(*,*) ' jackknife data file = ',jackfl(1:40)

      read(lin,*,err=98) ixlj,iylj,izlj,ivrlj
      write(*,*) ' columns = ',ixlj,iylj,izlj,ivrlj

      read(lin,*,err=98) ncut
      write(*,*) ' number of thresholds / categories = ',ncut
c
c Find the needed parameters:
c
      MAXCUT = ncut
      MAXROT = MAXNST*MAXCUT + 1
c
c Allocate the needed memory:
c1
      allocate(thres(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c2
      allocate(gcdf(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c3
      allocate(nst(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c4
      allocate(c0(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c5
      allocate(it(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c6
      allocate(cc(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c7
      allocate(aa(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c8
      allocate(ang1(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c9
      allocate(ang2(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c10
      allocate(ang3(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c11
      allocate(anis1(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c12
      allocate(anis2(MAXCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c13
      allocate(nviol(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c15
      allocate(ccdf(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c16
      allocate(ccdfo(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c17
      allocate(aviol(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c18
      allocate(xviol(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      read(lin,*,err=98) (thres(i),i=1,ncut)
      write(*,*) ' thresholds / categories = ',(thres(i),i=1,ncut)

      read(lin,*,err=98) (gcdf(i),i=1,ncut)
      write(*,*) ' global cdf / pdf        = ',(gcdf(i),i=1,ncut)

      read(lin,'(a512)',err=98) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=98) idhl,ixl,iyl,izl,ivrl
      write(*,*) ' columns = ',idhl,ixl,iyl,izl,ivrl

      read(lin,'(a512)',err=98) softfl
      call chknam(softfl,512)
      write(*,*) ' soft data file = ',softfl(1:40)
      inquire(file=softfl,exist=testfl)

      if(testfl) then
            read(lin,*,err=98) ixs,iys,izs,(ivrs(i),i=1,ncut)
            write(*,*) ' columns = ',ixs,iys,izs,(ivrs(i),i=1,ncut)
      else
            read(lin,*,err=98)
      end if

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=98) idbg
      write(*,*) ' debugging level = ',idbg

      read(lin,'(a512)',err=98) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debug file = ',dbgfl(1:40)
      open(ldbg,file=dbgfl,status='UNKNOWN')

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz = ',nx,xmn,xsiz

      read(lin,*,err=98) ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz = ',ny,ymn,ysiz

      read(lin,*,err=98) nz,zmn,zsiz
      write(*,*) ' nz, zmn, zsiz = ',nz,zmn,zsiz

      read(lin,*,err=98) ndmin,ndmax
      write(*,*) ' ndmin, ndmax = ',ndmin,ndmax

      read(lin,*,err=98) radius,radius1,radius2
      write(*,*) ' search radii = ',radius,radius1,radius2
      if(radius.lt.EPSLON) stop 'radius must be greater than zero'
      radsqd = radius  * radius
      sanis1 = radius1 / radius
      sanis2 = radius2 / radius

      read(lin,*,err=98) sang1,sang2,sang3
      write(*,*) ' search anisotropy angles = ',sang1,sang2,sang3

      read(lin,*,err=98) noct
      write(*,*) ' number per octant = ',noct

      read(lin,*,err=98) mik, cutmik
      write(*,*) ' median IK option = ',mik,cutmik

      read(lin,*,err=98) ktype
      write(*,*) ' ktype (0=SK, 1=OK) = ',ktype
c
c Find the needed paramters:
c
      MAXSBX = 1
      if(nx.gt.1)then
            MAXSBX = int(nx/2.00)
            if(MAXSBX.gt.50)MAXSBX=50
      end if
c
      MAXSBY = 1
      if(ny.gt.1)then
            MAXSBY = int(ny/2.00)
            if(MAXSBY.gt.50)MAXSBY=50
      end if
c
      MAXSBZ = 1
      if(nz.gt.1)then
            MAXSBZ = int(nz/2.00)
            if(MAXSBZ.gt.50)MAXSBZ=50
      end if
c
      MAXSB = MAXSBX*MAXSBY*MAXSBZ
      MAXSAM = ndmax + 1
      MAXEQ = MAXSAM + 1
c
c Allocate the needed memory:
c19
      allocate(nisb(MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c20
      allocate(ixsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c21
      allocate(iysbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c22
      allocate(izsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c23
      allocate(rotmat(MAXROT,3,3),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c31
      allocate(xa(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c32
      allocate(ya(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c33
      allocate(za(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c34
      allocate(vra(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c35
      allocate(r(MAXEQ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c36
      allocate(s(MAXEQ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c37
      allocate(a(MAXEQ*MAXEQ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
c Read all of the variograms and write the debugging the debug file:
c
      do i=1,ncut
            read(lin,*,err=98) nst(i),c0(i)
            if(ivtype.eq.0)
     +      write(ldbg,100)  i,thres(i),gcdf(i),nst(i),c0(i)
            if(ivtype.eq.1)
     +      write(ldbg,101)  i,thres(i),gcdf(i),nst(i),c0(i)
            if(nst(i).gt.MAXNST) stop 'nst is too big'
            istart = 1 + (i-1)*MAXNST
            do j=1,nst(i)
                  index = istart + j - 1
                  read(lin,*,err=98) it(index),cc(index),ang1(index),
     +                               ang2(index),ang3(index)
                  if(it(index).eq.3) STOP 'Gaussian Model Not Allowed!'
                  read(lin,*,err=98) aa(index),aa1,aa2
                  write(ldbg,102)  j,it(index),aa(index),cc(index)
                  anis1(index) = aa1 / aa(index)
                  anis2(index) = aa2 / aa(index)
                  write(ldbg,103) ang1(index),ang2(index),ang3(index),
     +                            anis1(index),anis2(index)
            end do
      end do
      close(lin)
 100  format(/,' Category  number ',i2,' = ',f12.3,/,
     +         '           global prob value = ',f8.4,/,
     +         '           number of structures = ',i3,/,
     +         '           nugget effect        = ',f8.4)
 101  format(/,' Threshold number ',i2,' = ',f12.3,/,
     +         '           global prob value = ',f8.4,/,
     +         '           number of structures = ',i3,/,
     +         '           nugget effect        = ',f8.4)
 102  format(  '           type of structure ',i3,' = ',i3,/,
     +         '           aa parameter         = ',f12.4,/,
     +         '           cc parameter         = ',f12.4)
 103  format(  '           ang1, ang2, ang3     = ',3f6.2,/,
     +         '           anis1, anis2         = ',2f12.4)
c
c Check to make sure the data file exists, then either read in the
c data or write an error message and stop:
c
      nd = 0
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) ' ERROR data file ',datafl,' does not exist!'
            stop
      end if
c
c Ppen the file and read in the header information. Initialize the
c storage that will be used to summarize the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*)
      end do
      MAXDAT = 0
 22   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      if(var(ivrl).lt.tmin.or.var(ivrl).ge.tmax)go to 22
      MAXDAT = MAXDAT + 1
      go to 22
 44   continue
c
c Allocate the needed memory:
c24
      allocate(x(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c25
      allocate(y(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c26
      allocate(z(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c27
      allocate(vr(MAXDAT,MAXCUT+1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c28
      allocate(tmp(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c29
      allocate(close(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c30
      allocate(actloc(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c38
      allocate(dh(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c14
      allocate(sb(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c39
      allocate(sdis(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a40)')  str
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*,err=99)
      end do
c
c Read all the data until the end of the file:
c
 2    read(lin,*,end=3,err=99) (var(j),j=1,nvari)
      vrt = var(ivrl)
      if(vrt.lt.tmin.or.vrt.ge.tmax) go to 2
      nd = nd + 1
      if(nd.gt.MAXDAT) then
            write(*,*) ' ERROR: Exceeded available memory for data'
            stop
      end if
      if(idhl.le.0) then
            dh(nd) = -99
      else
            dh(nd) = var(idhl)
      endif
      if(ixl.le.0) then
            x(nd)  = xmn
      else
            x(nd)  = var(ixl)
      endif
      if(iyl.le.0) then
            y(nd)  = ymn
      else
            y(nd)  = var(iyl)
      endif
      if(izl.le.0) then
            z(nd)  = zmn
      else
            z(nd)  = var(izl)
      endif
c
c The indicator data are constructed knowing the thresholds and the
c data value.
c
      vr(nd,ncut+1) = vrt
      if(ivtype.eq.0) then
            do ic=1,ncut
                  vr(nd,ic) = 0.0
                  if(int(vrt+0.5).eq.int(thres(ic)+0.5)) vr(nd,ic)=1.0
            end do
      else
            do ic=1,ncut
                  vr(nd,ic) = 1.0
                  if(vrt.gt.thres(ic)) vr(nd,ic) = 0.0
            end do
      end if
c
c Return for another data:
c
      go to 2
 3    close(lin)
c
c Compute the averages and variances as an error check for the user:
c
      av = av / max(real(nd),1.0)
      ss =(ss / max(real(nd),1.0)) - av * av
      write(*,*)
      write(*,*) 'Data for IK3D: Variable number ',ivrl
      write(*,*) '  Number   = ',nd
      ndh = nd
c
c Direct input of indicator data:
c
      inquire(file=softfl,exist=testfl)
      if(testfl) then
            write(*,*)
            write(*,*) 'Reading direct indicator data'
            open(lin,file=softfl,status='OLD')
            read(lin,*,err=97)
            read(lin,*,err=97) nvari
            if(nvari.ne.(ncut+3)) then
                  write(*,*) ' ERROR: number of variables in ',softfl
                  write(*,*) '        is inconsistent with ncut + 3'
                  stop
            end if
            do i=1,nvari
                  read(lin,*,err=97)
            end do
 12         read(lin,*,end=13,err=97) (var(j),j=1,nvari)
c
c Check for co-located:
c
            if(ixs.le.0) then
                  xx = xmn
            else
                  xx = var(ixs)
            endif
            if(iys.le.0) then
                  yy = ymn
            else
                  yy = var(iys)
            endif
            if(izs.le.0) then
                  zz = zmn
            else
                  zz = var(izs)
            endif
            do i=1,ndh
                  test = abs(xx-x(i)) + abs(yy-y(i)) + abs(zz-z(i))
                  if(test.le.EPSLON) go to 12
            end do
c
c Accept this data:
c
            nd     = nd + 1
            x(nd)  = xx
            y(nd)  = yy
            z(nd)  = zz
            do j=1,ncut
                  i = ivrs(j)
                  vr(nd,j) = var(i)
            end do
c
c If performing median IK then check for missing values:
c
            if(mik.eq.1) then
                  do ic=1,ncut
                        if(vr(nd,ic).lt.0.0) then
                              write(*,150) softfl
                              stop
                        endif
                  end do
 150              format(' Since the median IK approach is being',
     +                   ' considered no missing values are',
     +                   ' allowed',/,' Check file ',a40)
            endif
            go to 12 
 13         close(lin)
      endif
c
c Load the right variogram as the first one if performing median IK:
c
      if(mik.eq.1) then
            icut = 1
            clos = abs(cutmik-thres(1))
            do ic=2,ncut
                  test = abs(cutmik-thres(ic))
                  if(test.lt.clos) then
                        icut = ic
                        clos = test
                  end if
            end do
            c0(1)   = c0(icut)
            nst(1)  = nst(icut)
            istart1 = 1
            istarti = 1 + (icut-1)*MAXNST
            do ist=1,nst(1)
                  index1        = istart1 + ist - 1
                  indexi        = istarti + ist - 1
                  it(index1)    = it(indexi)
                  aa(index1)    = aa(indexi)
                  cc(index1)    = cc(indexi)
                  ang1(index1)  = ang1(indexi)
                  ang2(index1)  = ang2(indexi)
                  ang3(index1)  = ang3(indexi)
                  anis1(index1) = anis1(indexi)
                  anis2(index1) = anis2(indexi)
            end do
      end if
c
c Open the output file and write a header:
c
      nreal = 1
      open(lout,file=outfl,status='UNKNOWN')
      if(koption.eq.0) then
            write(lout,200) str
            write(lout,204) ncut,nx,ny,nz,
     +                      xmn,ymn,zmn,xsiz,ysiz,zsiz,nreal      
      else
            write(lout,200) str
            write(lout,204) ncut+1,nx,ny,nz,
     +                      xmn,ymn,zmn,xsiz,ysiz,zsiz,nreal            
      end if
 200  format('IK3D Estimates with:',a40)
 204  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      do i=1,ncut
            if(ivtype.eq.0) write(lout,201) i,thres(i)
            if(ivtype.eq.1) write(lout,202) i,thres(i)
 201        format('Category:  ',i2,' = ',g14.8)
 202        format('Threshold: ',i2,' = ',g14.8)
      end do
      if(koption.gt.0) write(lout,203)
 203  format('true value')
c
c Set up for cross validation:
c
      if(koption.eq.1) then
            jackfl = datafl
            idhlj  = idhl
            ixlj   = ixl
            iylj   = iyl
            izlj   = izl
            ivrlj  = ivrl
      end if
c
c Open the file with the jackknife data?
c
      if(koption.gt.0) then
              write(*,*)
                  write(*,*) ' Opening ',jackfl(1:20),' for validation'
                  write(*,*)
            inquire(file=jackfl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',jackfl,' does not exist!'
                  stop
            endif
            open(ljack,file=jackfl,status='OLD')
            read(ljack,*,err=96)
            read(ljack,*,err=96) nvarij
            do i=1,nvarij
                  read(ljack,*,err=96)
            end do
      end if
c
c
c
      return
c
c Error in an Input File Somewhere:
c
 96   stop 'ERROR in jackknife file!'
 97   stop 'ERROR in soft data file!'
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine ik3d(MAXCUT,MAXROT,MAXSBX,MAXSBY,MAXSBZ)
c-----------------------------------------------------------------------
c
c                   Multiple Indicator Kriging
c                   **************************
c
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'ik3d.inc'
      integer    infoct(8)
      real       UNEST,var(100)
      logical    krig,accept
      data       UNEST/-9.9999/
c
c Set up the rotation/anisotropy matrices that are needed for the
c variogram and search:
c
      write(*,*) 'Setting up rotation matrices for variogram and search'
      radsqd = radius * radius
      do ic=1,ncut
      do is=1,nst(ic)
            ind = is + (ic-1)*MAXNST
            call setrot(ang1(ind),ang2(ind),ang3(ind),anis1(ind),
     +                  anis2(ind),ind,MAXROT,rotmat)
      end do
      end do
      isrot = MAXNST*MAXCUT + 1
      call setrot(sang1,sang2,sang3,sanis1,sanis2,isrot,MAXROT,rotmat)
c
c Set up for super block searching:
c
      do i=1,nd
            actloc(i) = real(i)
      end do
      write(*,*) 'Setting up super block search strategy'
      nsec = 0
      call setsupr(nx,xmn,xsiz,ny,ymn,ysiz,nz,zmn,zsiz,nd,x,y,z,actloc,
     +             tmp,nsec,sec1,sec2,sec3,MAXSBX,MAXSBY,MAXSBZ,nisb,
     +             nxsup,xmnsup,xsizsup,nysup,ymnsup,ysizsup,nzsup,
     +             zmnsup,zsizsup)
      call picksup(nxsup,xsizsup,nysup,ysizsup,nzsup,zsizsup,
     +             isrot,MAXROT,rotmat,radsqd,nsbtosr,ixsbtosr,
     +             iysbtosr,izsbtosr)
c
c Initialize accumulators:
c
      nk = 0
      xk = 0.0
      vk = 0.0
      do icut=1,ncut
            nviol(icut) =  0
            aviol(icut) =  0.0
            xviol(icut) = -1.0
      end do
      nxy   = nx*ny
      nxyz  = nx*ny*nz
      write(*,*)
      write(*,*) 'Working on the kriging '
c
c Set up for cross validation:
c
c
c Report on progress from time to time:
c
      if(koption.eq.0) then
            nxy   = nx*ny
            nxyz  = nx*ny*nz
            nloop = nxyz
            irepo = max(1,min((nxyz/10),10000))
      else
            nloop = 10000000
            irepo = max(1,min((nd/10),10000))
      end if
      ddh = 0.0
c
c MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
c
      do index=1,nloop
      if((int(index/irepo)*irepo).eq.index) write(*,103) index
 103  format('   currently on estimate ',i9)
c
c Where are we making an estimate?
c
      if(koption.eq.0) then
            iz   = int((index-1)/nxy) + 1
            iy   = int((index-(iz-1)*nxy-1)/nx) + 1
            ix   = index - (iz-1)*nxy - (iy-1)*nx
            xloc = xmn + real(ix-1)*xsiz
            yloc = ymn + real(iy-1)*ysiz
            zloc = zmn + real(iz-1)*zsiz
      else
            ddh = 0.0
            read(ljack,*,err=96,end=22) (var(i),i=1,nvarij)
            xloc = xmn
            yloc = ymn
            zloc = zmn
            true = UNEST
            if(idhlj.gt.0)  ddh    = var(idhlj)
            if(ixlj.gt.0)   xloc   = var(ixlj)
            if(iylj.gt.0)   yloc   = var(iylj)
            if(izlj.gt.0)   zloc   = var(izlj)
            if(ivrlj.gt.0)  true   = var(ivrlj)
      end if
c
c Find the nearest samples:
c
            call srchsupr(xloc,yloc,zloc,radsqd,isrot,MAXROT,rotmat,
     +                    nsbtosr,ixsbtosr,iysbtosr,izsbtosr,noct,
     +                    nd,x,y,z,tmp,nisb,nxsup,xmnsup,xsizsup,nysup,
     +                    ymnsup,ysizsup,nzsup,zmnsup,zsizsup,nclose,
     +                    close,infoct)
c
c Test number of samples found and octants informed:
c
            if(nclose.lt.ndmin) then
                  if(idbg.ge.2) write(ldbg,*) 'Too few data:',ix,iy,iz
                  do i=1,ncut
                        ccdfo(i) = UNEST
                  end do
                  go to 1
            endif
c
c Loop over all the thresholds/categories:
c
            do 2 ic=1,ncut
                  krig = .true.
                  if(mik.eq.1.and.ic.ge.2) krig = .false.
c
c Identify the close data (there may be a different number of data at
c each threshold because of constraint intervals); however, if
c there are no constraint intervals then this step can be avoided.
c
                  nca = 0
                  do ia=1,nclose
                        j  = int(close(ia)+0.5)
                        ii = actloc(j)
                        accept = .true.
                        if(koption.ne.0.and.(abs(x(j)-xloc)+
     +                     abs(y(j)-yloc)+ abs(z(j)-zloc)).lt.EPSLON)
     +                                        accept = .false.
                        if(koption.ne.0.and.(abs(dh(ii)-ddh))
     +                        .lt.EPSLON)     accept = .false.
                        if(vr(ii,ic).lt.tmin) accept = .false.
                        if(vr(ii,ic).gt.tmax) accept = .false.
                        if(accept) then
                              nca = nca + 1
                              vra(nca) = vr(ii,ic)
                              xa(nca)  = x(j)
                              ya(nca)  = y(j)
                              za(nca)  = z(j)
                        endif
                        if(nca.eq.ndmax) go to 3
                  end do
 3                continue
c
c If there are no samples at this threshold then use the global cdf:
c
                  if(nca.eq.0) then
                        ccdf(ic) = gcdf(ic)
                        go to 2
                  endif
c
c Now, only load the variogram, build the matrix,... if kriging:
c
                  if(krig) then
                  neq   = nca + ktype
c
c Solve the Kriging System with more than one sample:
c
                  in   = 0
                  irot = 1 + (ic-1)*MAXNST
                  do j=1,nca
                        do i=1,j
                              in  = in + 1
                              call cova3(xa(i),ya(i),za(i),xa(j),ya(j),
     +                             za(j),ic,nst,MAXNST,c0,it,cc,aa,irot,
     +                             MAXROT,rotmat,cmax,cov)
                              a(in) = dble(cov)
                        end do
                        call cova3(xa(j),ya(j),za(j),xloc,yloc,
     +                             zloc,ic,nst,MAXNST,c0,it,cc,aa,irot,
     +                             MAXROT,rotmat,cmax,cov)
                        r(j)  = dble(cov)
                  end do
c
c Ordinary Kriging unbiasedness constraint:
c
                  if(ktype.eq.1) then
                        do i=1,nca
                              in    = in + 1
                              a(in) = 1.0
                        end do
                        in      = in + 1
                        a(in)   = 0.0
                        r(neq)  = 1.0
                  endif
c
c Write out the kriging Matrix if Seriously Debugging:
c
                  if(idbg.ge.3) then
                        write(ldbg,101) ix,iy,iz
                        is = 1
                        do i=1,neq
                              ie = is + i - 1
                              write(ldbg,102) i,r(i),(a(j),j=is,ie)
                              is = is + i
                        end do
 101                    format(/,'Kriging Matrices for Node: ',3i4)
 102                    format('    r(',i2,') =',f7.4,'  a= ',9(10f7.4))
                  endif
c
c Solve the system:
c
                  if(neq.eq.1) then
                        ising = 0.0
                        s(1)  = r(1) / a(1)
                  else
                        call ksol(1,neq,1,a,r,s,ising)
                  end if
c
c More Debugging Information:
c
                  if(idbg.eq.3) then
                        do k=1,nca
                        write(ldbg,98) xa(k),ya(k),za(k),vra(k),s(k)
 98                     format('Loc: x y z ',3f9.1,' val wt ',2f12.5)
                        end do
                  endif
c
c Compute the solution if not singular:
c
                  if(ising.ne.0) then
                        write(ldbg,*) 'Singular at ',ix,iy,iz,ic
                        if(idbg.ge.3) stop
                        do i=1,ncut
                              ccdfo(i) = UNEST
                        end do
                        go to 1
                  endif
c
c Finished kriging (if it was necessary):
c
                  end if
c
c Compute Kriged estimate of cumulative probability:
c
                  sumwts   = 0.0
                  ccdf(ic) = 0.0
                  do i=1,nca
                        ccdf(ic) = ccdf(ic) + vra(i)*real(s(i))
                        sumwts   = sumwts   + real(s(i))
                  end do
                  if(ktype.eq.0) 
     +            ccdf(ic) = ccdf(ic) + (1.0-sumwts)*gcdf(ic)
c
c Keep looping until all the thresholds are estimated:
c 
 2          continue
c
c Correct and write the distribution to the output file:
c
            nk = nk + 1
            call ordrel(ivtype,ncut,ccdf,ccdfo,nviol,aviol,xviol)
c
c Debugging information:
c
            if(idbg.ge.3) then
                  write(ldbg,104) (ccdf(i),i=1,ncut)
                  write(ldbg,105) (ccdfo(i),i=1,ncut)
 104              format('Uncorrected: ',30(f8.4))
 105              format('Corrected:   ',30(f8.4))
            endif
c
c Write the IK CCDF for this grid node:
c
 1    continue
      if(koption.eq.0) then
            write(lout,'(30(f8.4))') (ccdfo(i),i=1,ncut)
      else
            write(lout,'(30(f8.4))') (ccdfo(i),i=1,ncut),true
      end if
c
c END OF MAIN KRIGING LOOP:
c
      end do
c
c Write summary of order relations corrections:
c
 22   continue
      ntot = 0
      atot = 0.0
      write(ldbg,300) 
 300  format(/,' Summary of order relations (number and magnitude): ')
      do icut=1,ncut
            ntot = ntot + nviol(icut)
            atot = atot + aviol(icut)
            aviol(icut) = aviol(icut) / real(max(1,nviol(icut)))
            if(ivtype.eq.0)
     +      write(ldbg,301) icut,nviol(icut),aviol(icut),xviol(icut)
            if(ivtype.eq.1)
     +      write(ldbg,302) icut,nviol(icut),aviol(icut),xviol(icut)
 301        format('   Category ',i2,' Number = ',i6,' Average = ',f8.4,
     +             ' Maximum = ',f8.4)
 302        format('   Threshold',i2,' Number = ',i6,' Average = ',f8.4,
     +             ' Maximum = ',f8.4)
      end do
      atot = atot / real(max(1,ntot))
      btot =(ntot / max(1.0,real(ncut*nk))) * 100.0
      write(ldbg,303) btot,atot
 303  format(/,' Total of ',f7.2,'% with an average magnitude of ',f8.4)
      write(*,*)
      write(*,*)' Finished kriging ',nk,' locations'
c
c All finished the kriging:
c
      return
 96   stop 'ERROR in jackknife file'
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
      open(lun,file='ik3d.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for IK3D',/,
     +       '                  *******************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('1                                ',
     +       '-1=continuous(cdf), 0=categorical(pdf)')
      write(lun,12)
 12   format('0                                ',
     +       '-option: 0=grid, 1=cross, 2=jackknife')
      write(lun,13)
 13   format('jack.dat                         ',
     +       '-file with jackknife data')
      write(lun,14)
 14   format('1   2   0    3                   ',
     +       '-   columns for X,Y,Z,vr')
      write(lun,15)
 15   format('5                                ',
     +       '-number thresholds/categories')
      write(lun,16)
 16   format('0.5   1.0   2.5   5.0   10.0     ',
     +       '-   thresholds / categories')
      write(lun,17)
 17   format('0.12  0.29  0.50  0.74  0.88     ',
     +       '-   global cdf / pdf')
      write(lun,18)
 18   format('../data/cluster.dat              ',
     +       '-file with data')
      write(lun,19)
 19   format('0   1   2   0    3               ',
     +       '-   columns for DH,X,Y,Z and variable')
      write(lun,20)
 20   format('direct.ik                        ',
     +       '-file with soft indicator input')
      write(lun,21)
 21   format('1   2   0    3  4  5  6          ',
     +       '-   columns for X,Y,Z and indicators')
      write(lun,22)
 22   format('-1.0e21   1.0e21                 ',
     +       '-trimming limits')
      write(lun,23)
 23   format('2                                ',
     +       '-debugging level: 0,1,2,3')
      write(lun,24)
 24   format('ik3d.dbg                         ',
     +       '-file for debugging output')
      write(lun,25)
 25   format('ik3d.out                         ',
     +       '-file for kriging output')
      write(lun,26)
 26   format('10   2.5    5.0                  ',
     +       '-nx,xmn,xsiz')
      write(lun,27)
 27   format('10   2.5    5.0                  ',
     +       '-ny,ymn,ysiz')
      write(lun,28)
 28   format('1    0.0    5.0                  ',
     +       '-nz,zmn,zsiz')
      write(lun,29)
 29   format('1    8                           ',
     +       '-min, max data for kriging')
      write(lun,30)
 30   format('20.0  20.0  20.0                 ',
     +       '-maximum search radii')
      write(lun,31)
 31   format(' 0.0   0.0   0.0                 ',
     +       '-angles for search ellipsoid')
      write(lun,32)
 32   format('0                                ',
     +       '-max per octant (0-> not used)')
      write(lun,33)
 33   format('1   2.5                          ',
     +       '-0=full IK, 1=Median IK(threshold num)')
      write(lun,34)
 34   format('1                                ',
     +       '-0=SK, 1=OK')
      write(lun,35)
 35   format('1    0.15                        ',
     +       '-One   nst, nugget effect')
      write(lun,36)
 36   format('1    0.85 0.0   0.0   0.0        ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,37)
 37   format('         10.0  10.0  10.0        ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,38)
 38   format('1    0.1                         ',
     +       '-Two   nst, nugget effect')
      write(lun,39)
 39   format('1    0.9  0.0   0.0   0.0        ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,40)
 40   format('         10.0  10.0  10.0        ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,41)
 41   format('1    0.1                         ',
     +       '-Three nst, nugget effect')
      write(lun,42)
 42   format('1    0.9  0.0   0.0   0.0        ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,43)
 43   format('         10.0  10.0  10.0        ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,44)
 44   format('1    0.1                         ',
     +       '-Four  nst, nugget effect')
      write(lun,45)
 45   format('1    0.9  0.0   0.0   0.0        ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,46)
 46   format('         10.0  10.0  10.0        ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,47)
 47   format('1    0.15                        ',
     +       '-Five  nst, nugget effect')
      write(lun,48)
 48   format('1    0.85 0.0   0.0   0.0        ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,49)
 49   format('         10.0  10.0  10.0        ',
     +       '-      a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
