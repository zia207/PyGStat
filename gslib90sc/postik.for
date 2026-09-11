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
c                   Post Process IK Distributions
c                   *****************************
c
c Reads IK-generated distributions and post processes them
c according to user specifications.
c
c      - affine or indirect lognormal volume support correction
c      - E-type mean
c      - probability of exceeding specified threshold and the mean
c           value above threshold.
c      - compute a z-threshold value for a specified CDF value.
c
c See the text for detailed discussion of parameters to extrapolate the
c discretely coded cdf values.
c
c
c INPUT/OUTPUT Parameters:
c
c   distin           the input distributions (output from IK3D)
c   outfl            the output file for E-type,....
c   iout,outpar         =1 E-type,
c                       =2 prob and grade > and <= outpar; 
c                       =3 Z percentile corresponding to outpar,
c                       =4 conditional variance.
c   nccut            the number of cutoffs
c   ccut(i)          cutoffs
c   ivol             =1, the consider volume support corrections
c   ivtyp            =1, affine; =2 indirect lognormal
c   varred           variance adjustment factor, between 0 and 1
c   datafl           the global data distribution
c   ivr,iwt,tmin     parameters to read in global data distribution
c   varred           variance reduction parameter
c   zmin,zmax        minimum and maximum Z values
c   ltail,ltpar      option and parameter to handle values in lower tail
c   middle,mpar      option and parameter to handle values in the middle
c   utail,utpar      option and parameter to handle values in upper tail
c   maxdis           discretization to compute E-type and mean > cutoff
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
c
c User adjustable parameters:
c
      parameter(MAXCUT = 100)
c
c Fixed parameters:
c
      parameter(MV=500,UNEST=-99.,EPSLON=1.0e-6,VERSION=3.000)
      
      character distin*512,datafl*512,outfl*512,str*512
      real      ccdf(MAXCUT),ccdf1(MAXCUT),ccdf2(MAXCUT),ccut(MAXCUT),
     +          ccut1(MAXCUT),val(MV),ltpar,mpar,utpar,var(50)
      integer   ltail,middle,utail,test
      logical   testfl
      data      lin/1/,lout/2/
c
c  Dynamic allocation of arrays that depend on the number of data:
c
      real, allocatable :: cdf(:),cut(:)
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' POSTIK Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'postik.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'postik.par          ') then
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

      read(lin,'(a512)',err=97) distin
      call chknam(distin,512)
      write(*,*) ' input distributions = ',distin(1:40)

      read(lin,'(a512)',err=97) outfl 
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) iout,outpar
      write(*,*) ' output option & par = ',iout,outpar
      if(iout.ne.1.and.iout.ne.2.and.iout.ne.3.and.iout.ne.4) then
            write(*,*) ' ERROR: invalid output option ',iout
            stop
      end if
      if(iout.eq.3) then
            if(outpar.lt.0.0) stop 'Invalid p-value for iout=3'
            if(outpar.gt.1.0) stop 'Invalid p-value for iout=3'
            outpar = min(max(outpar,EPSLON),(1.0-EPSLON))
      end if

      read(lin,*,err=97) nccut
      write(*,*) ' number of cutoffs = ',nccut
      if(nccut.gt.MAXCUT) stop 'ncut is too big - modify parameters'

      read(lin,*,err=97) (ccut1(i),i=1,nccut)
      write(*,*) ' cutoffs = ',(ccut1(i),i=1,nccut)

      read(lin,*,err=97) ivol,ivtyp,varred
      write(*,*) ' volume variance = ',ivol,ivtyp,varred
      if(varred.lt.0.0.or.varred.gt.1.0) then
            write(*,*) ' ERROR: invalid variance reduction ',varred
            write(*,*) '        must be between 0 and 1'
            stop
      end if

      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' tabulated quantiles = ',datafl(1:40)

      read(lin,*,err=97) ivr,iwt,tmin,tmax
      write(*,*) ' input columns...     ',ivr,iwt,tmin,tmax

      read(lin,*,err=97) zmin,zmax
      write(*,*) ' minimum and maximum = ',zmin,zmax
      if(iout.eq.2.and.outpar.lt.zmin) stop 'Invalid z-value for iout=2'
      if(iout.eq.2.and.outpar.gt.zmax) stop 'Invalid z-value for iout=2'

      read(lin,*,err=97) ltail,ltpar
      write(*,*) ' ltail, ltpar = ',ltail,ltpar

      read(lin,*,err=97) middle,mpar
      write(*,*) ' middle, mpar = ',middle,mpar

      read(lin,*,err=97) utail,utpar
      write(*,*) ' utail, utpar = ',utail, utpar

      read(lin,*,err=97) maxdis
      write(*,*) ' discretization = ',maxdis

      close(lin)
c
c Do we have a global distribution and do we need it anyway?
c
      ncut = 0
      inquire(file=datafl,exist=testfl)
      if(ltail.ne.3.and.middle.ne.3.and.utail.ne.3) testfl = .false.
c
c Read in the global cdf ("cut" and "cdf" arrays):
c
      if(testfl) then
            tcdf = 0.0
c
c The first data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
            open(lin,file=datafl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*,err=98)
            end do
            maxdat = 0
 22         read(lin,*,end=33,err=99)(var(j),j=1,nvari)
            if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 22
            maxdat = maxdat + 1
            go to 22
 33         continue
            
c
c  Now allocate the needed memory:
c
            allocate (cdf(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                  'insufficient memory!', test
                  stop
            end if
c
            allocate (cut(maxdat),stat = test)
            if (test.ne.0) then
                 write(*,*) 'Error: Allocation failed due to ',
     +                      'insufficient memory!', test
                 stop
            end if
            rewind(lin)
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*,err=98)
            end do
            gmean = 0.0
            ncut  = 0
 5          read(lin,*,end=6,err=98) (val(j),j=1,nvari)
            if(val(ivr).lt.tmin.or.val(ivr).ge.tmax) go to 5
            if(iwt.ge.1) then
                  if(val(iwt).le.0.0) go to 5
            endif
            ncut = ncut + 1
            if(ncut.gt.MAXDAT) then
                  write(*,*) ' ERROR: exceeded MAXDAT'
                  stop
            endif
            cut(ncut) = val(ivr)
            gmean     = gmean + val(ivr)
            if(iwt.le.0) then
                  cdf(ncut) = 1.0
            else
                  cdf(ncut) = val(iwt)
            endif
            tcdf = tcdf + cdf(ncut)
            go to 5
 6          close(lin)
            if(tcdf.le.0) stop ' total global CDF <= 0'
            gmean = gmean / tcdf
c
c Turn the (possibly weighted) distribution into a cdf that is useful:
c
            call sortem(1,ncut,cut,1,cdf,c,d,e,f,g,h)
            oldcp = 0.0
            cp    = 0.0
            tcdf  = 1.0 / tcdf
            do i=1,ncut
                  cp     = cp + cdf(i) * tcdf
                  cdf(i) =(cp + oldcp) * 0.5
                  oldcp  = cp
            end do
c
c Get median and write some info to the screen:
c
            call locate(cdf,ncut,1,ncut,0.5,j)
            gmedian = powint(cdf(j),cdf(j+1),cut(j),cut(j+1),0.5,1.)
            write(*,*) 'Global cdf from file: ',datafl
            write(*,*) '   number of data: ',ncut
            write(*,*) '   global mean:    ',gmean
            write(*,*) '   global median:  ',gmedian
            write(*,*)
      endif
c
c Open the output file:
c
      open(lout,file=outfl,status='UNKNOWN')
      if(iout.eq.1) write(lout,100)
      if(iout.eq.2) write(lout,101) outpar
      if(iout.eq.3) write(lout,102) outpar
      if(iout.eq.4) write(lout,103)
 100  format('E-type mean values',/,'1',/,'mean')
 101  format('Probability and mean value > ',f12.4,/,
     +       '3',/,'prob > cutoff',/,'mean > cutoff',/,'mean < cutoff')
 102  format('Z value corresponding to CDF = ',f7.4,/,'1',/,'value')
 103  format('Conditional Variance',/,'1',/,'condv')
c
c Open and start reading through the IK distributions:
c
      inquire(file=distin,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR ',distin,' does not exist!'
            write(*,*) '  you need some distributions to work with'
            stop
      endif
      open(lin,file=distin,status='OLD')
      read(lin,*,err=96)
      read(lin,*,err=96) nvari
      do i=1,nvari
            read(lin,*,err=96)
      end do
c
c If we are doing the affine correction then compute the square root
c of the variance ``adjustment'' factor:
c
      if(ivol.eq.1.and.ivtyp.eq.1) varred = sqrt(max(varred,0.0))
c
c BIG LOOP reading in each distribution in turn:
c
      nproc = 0
      procm = 0.0
 9    read(lin,*,end=99,err=96) (ccdf(i),i=1,nccut)
c
c Check for missing values:
c
      if(ccdf(nccut).lt.-0.1) then
            if(iout.eq.1) write(lout,110) UNEST
            if(iout.eq.2) write(lout,110) UNEST,UNEST,UNEST
            if(iout.eq.3) write(lout,110) UNEST
            if(iout.eq.4) write(lout,110) UNEST
 110        format(3f12.4)
            go to 9
      end if
c
c Reinstate "ccut1" because volume support may have changed "ccut":
c
      do i=1,nccut
            ccut(i) = ccut1(i)
      end do
c
c Correct Order Relations (the distributions coming from IK3D have
c already been corrected):
c
      do i=1,nccut
            ccdf(i)=max(min(1.0,ccdf(i)),0.0)
      end do
      ccdf1(1) = ccdf(1)
      do i=2,nccut
            ccdf1(i) = max(ccdf1(i-1),ccdf(i))
      end do
      ccdf2(nccut) = ccdf(nccut)
      do i=nccut-1,1,-1
            ccdf2(i) = min(ccdf2(i+1),ccdf(i))
      end do
      do i=1,nccut
            ccdf(i) = 0.5*(ccdf1(i)+ccdf2(i))
      end do
c
c Volume support correction with parameters (ivol,ivtyp,varred):
c
      if(ivol.eq.1) then
            dis    = 1.0 / real(maxdis)
            cdfval = -0.5*dis
            etype  = 0.0
            ecv    = 0.0
            do i=1,maxdis
                  cdfval  = cdfval + dis
                  zval    = -1.0
                  call beyond(1,nccut,ccut,ccdf,ncut,cut,cdf,zmin,
     +                        zmax,ltail,ltpar,middle,mpar,utail,
     +                        utpar,zval,cdfval,ierr)
                  if(ierr.ne.0) write(*,*) 'ERROR: ',ierr,' continuing'
                  etype = etype + zval
                  ecv   = ecv   + zval*zval
            end do
            etype = etype / real(maxdis)
            ecv   = sqrt(max((ecv/real(maxdis)-etype*etype),0.0))
     +                      / max(etype,EPSLON)
            if(etype.eq.0.0) then
                  write(*,*) 'NO support correction with 0 mean'
                  etype = EPSLON
            endif
c
c Affine Correction:
c
            if(ivtyp.eq.1) then
                  do i=1,nccut
                        ccut(i) = etype+varred*(ccut(i)-etype)
                  end do
            else
c
c Indirect lognormal:  1. Compute parameters "a" and "b"
c                      2. Correct quantiles
c                      3. recompute etype
c                      4. Recorrect quantiles to correct mean
c
c
c            Parameters "a" and "b":
c
                  b = sqrt(max((alog(varred*ecv*ecv+1)
     +                       /alog(ecv*ecv+1)),0.0))
                  a = (etype/sqrt(max((varred*ecv*ecv+1),0.0))) * 
     +                      (sqrt(max((ecv*ecv+1),0.0))/etype)**b
c
c            Correct quantiles:
c
                  do i=1,nccut
                        ccut(i) = a*ccut(i)**b
                  end do
c
c            New etype:
c
                  cdfval = -0.5*dis
                  enew  = 0.0
                  do i=1,maxdis
                        cdfval  = cdfval + dis
                        zval    = -1.0
                        call beyond(1,nccut,ccut,ccdf,ncut,cut,cdf,zmin,
     +                              zmax,ltail,ltpar,middle,mpar,utail,
     +                              utpar,zval,cdfval,ierr)
                        if(ierr.ne.0) then
                            write(*,*) 'ERROR: ',ierr,' continuing'
                        endif
                        enew = enew + zval
                  end do
                  enew = enew / real(maxdis)
c
c            Recorrect "quantiles":
c
                  do i=1,nccut
                        ccut(i) = (etype/enew)*ccut(i)
                  end do
            endif
      endif
c
c Compute mean of local distribution if E-type (iout=1), or need the
c mean above (below) a cutoff, or if performing volume support:
c
      if(iout.le.2.or.iout.eq.4) then
            dis    = 1.0 / real(maxdis)
            cdfval = -0.5*dis
            eabove = 0.0
            nabove = 0
            ebelow = 0.0
            nbelow = 0
            etype  = 0.0
            ecv    = 0.0
            do i=1,maxdis
                  cdfval  = cdfval + dis
                  zval    = -1.0
                  call beyond(1,nccut,ccut,ccdf,ncut,cut,cdf,zmin,
     +                              zmax,ltail,ltpar,middle,mpar,utail,
     +                              utpar,zval,cdfval,ierr)
                  if(ierr.ne.0) then
                        write(*,*) 'ERROR: ',ierr,' continuing'
                  endif
                  etype = etype + zval
                  ecv   = ecv   + zval*zval
                  if(zval.le.outpar) then
                        nbelow = nbelow + 1
                        ebelow = ebelow + zval
                  else
                        nabove = nabove + 1
                        eabove = eabove + zval
                  end if
            end do
c
c e-type and conditional variance:
c
            etype = etype / real(maxdis)
            ecv   = ecv/real(maxdis)-etype*etype
      endif
c
c Write out E-type?
c
      if(iout.eq.1) then
            write(lout,202) etype
 202        format(f12.4)
      endif
c
c Do we need probability and mean value above a threshold?
c
      if(iout.eq.2) then
c
c      Get probability:
c
            cdfval  = -1.0
            call beyond(1,nccut,ccut,ccdf,ncut,cut,cdf,zmin,
     +                  zmax,ltail,ltpar,middle,mpar,utail,
     +                  utpar,outpar,cdfval,ierr)
            if(ierr.ne.0) then
                  write(*,*) 'ERROR: ',ierr,' continuing'
            endif
            prob   = 1.0 - cdfval
            eabove = eabove / real(max(1,nabove))
            ebelow = ebelow / real(max(1,nbelow))
            write(lout,203) prob,eabove,ebelow
 203        format(f7.4,1x,f12.4,1x,f12.4)
      endif
c
c Do we need the "Z" value corresponding to a particular CDF?
c
      if(iout.eq.3) then
            zval    = -1.0
            call beyond(1,nccut,ccut,ccdf,ncut,cut,cdf,zmin,
     +                  zmax,ltail,ltpar,middle,mpar,utail,
     +                  utpar,zval,outpar,ierr)
            if(ierr.ne.0) then
                  write(*,*) 'ERROR: ',ierr,' continuing'
            endif
            write(lout,204) zval
 204        format(f12.4)
      endif
c
c Write out conditional variance:
c
      if(iout.eq.4) write(lout,'(f12.4)') ecv
c
c Return for another:
c
      nproc = nproc + 1
      procm = procm + etype
      go to 9
c
c Finished:
c
 99   procm = procm / max(real(nproc),1.0)
      write(*,*)
      write(*,*) 'Number of distributions ',nproc
      if(procm.ne.0.0) write(*,*) 'Overall mean:           ',procm
      if(iout.eq.1) write(*,*) 'Local Means (E-type):   ',outfl
      if(iout.eq.2) write(*,*) 'Prob and mean > cutoff: ',outfl
      if(iout.eq.3) write(*,*) 'Z values for outpar in: ',outfl
      write(*,*)
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' POSTIK Version: ',f5.3, ' Finished'/)
      stop
 96   stop 'ERROR in distribution (IK) file!'
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in global data file!'
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
      open(lun,file='postik.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for POSTIK',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../ik3d/ik3d.out                 ',
     +       '-file with IK3D output (continuous)')
      write(lun,12)
 12   format('postik.out                       ',
     +       '-file for output')
      write(lun,13)
 13   format('2   0.25                         ',
     +       '-output option, output parameter')
      write(lun,14)
 14   format('5                                ',
     +       '-number of thresholds')
      write(lun,15)
 15   format('0.5  1.0  2.5  5.0  10.0         ',
     +       '-the thresholds')
      write(lun,16)
 16   format('0   1      0.75                  ',
     +       '-volume support?, type, varred')
      write(lun,17)
 17   format('cluster.dat                      ',
     +       '-file with global distribution')
      write(lun,18)
 18   format('3   0    -1.0   1.0e21           ',
     +       '-   ivr,  iwt,  tmin,  tmax')
      write(lun,19)
 19   format('0.0    30.0                      ',
     +       '-minimum and maximum Z value')
      write(lun,20)
 20   format('1   1.0                          ',
     +       '-lower tail: option, parameter')
      write(lun,21)
 21   format('1   1.0                          ',
     +       '-middle    : option, parameter')
      write(lun,22)
 22   format('1   2.0                          ',
     +       '-upper tail: option, parameter')
      write(lun,23)
 23   format('100                              ',
     +       '-maximum discretization')
      write(lun,24)
 24   format(//'option 1 = E-type',/,
     +         '       2 = probability and mean above threshold(par)',/,
     +         '       3 = Z percentile corresponding to (par)'/,
     +         '       4 = conditional variance')

      close(lun)
      return
      end
