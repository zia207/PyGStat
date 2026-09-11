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
c                Post Process Simulated Realizations
c                ***********************************
c
c Reads in a set of simulated realizations and post processes them
c according to user specifications.
c
c      1. Computes the E-type mean
c      2. Given a Z-cutoff the program will compute the probability of
c         exceeding the cutoff and the mean value above (and below)
c         the cutoff.
c      3. Given a CDF value the program will compute the corresponding
c         Z-percentile.
c      4. symmetric "P" probability interval
c      5. conditional variances
c
c
c INPUT/OUTPUT Parameters:
c
c      datafl         input realizations
c      outfl          output summary
c      tmin           missing value code
c      iout,outpar    output option and parameter
c      nx,ny,nz       the number of nodes in each coordinate direction
c      nsim           the number of realizations in datafl
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(EPSLON=1.e-12,UNEST=-999.,VERSION=3.000)
      character datafl*512,outfl*512,str*512
      real      meana,meanb
      integer   test
      logical   testfl
      data      lin/1/,lout/2/
c
c Declare dynamic arrays:
c
      real,allocatable :: var(:,:,:),cut(:),cdf(:)
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' POSTSIM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'postsim.par         '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'postsim.par         ') then
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
      write(*,*) ' data file with simulations = ',datafl(1:40)

      read(lin,*,err=97) nsim
      write(*,*) ' number of simulations = ',nsim

      read(lin,*,err=97) tmin
      write(*,*) ' lower trimming limit = ',tmin

      read(lin,*,err=97) nx,ny,nz
      write(*,*) ' grid size (nx,ny,nz) = ',nx,ny,nz
c
c Find the parameters, then allocate the needed memory.
c
      allocate(var(nx,ny,nsim),stat = test)
      if(test.ne.0)then
            write(*,*)'ERROR: Allocation failed due to',
     +                ' insufficient memory.',test
            stop
      end if

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) iout,outpar
      write(*,*) ' output option and parameter:',iout,outpar

      if(iout.eq.3) then
            if(outpar.lt.0.0) stop 'Invalid p-value for iout=3'
            if(outpar.gt.1.0) stop 'Invalid p-value for iout=3'
      end if

      close(lin)
c
c Allocation:
c
      allocate(cdf(nsim),stat = test)
      if(test.ne.0)then
            write(*,*)'ERROR: Allocation failed due to',
     +                ' insufficient memory.',test
            stop
      end if
      allocate(cut(nsim),stat = test)
      if(test.ne.0)then
            write(*,*)'ERROR: Allocation failed due to',
     +                ' insufficient memory.',test
            stop
      end if
c
c Set up cdf once:
c
      cdfinc = 1.0/real(nsim)
      cdf(1) = 0.5*cdfinc
      do i=2,nsim
            cdf(i) = cdf(i-1) + cdfinc
      end do
      if(iout.eq.3) then
            if(outpar.lt.cdf(1)   ) outpar = cdf(1)
            if(outpar.gt.cdf(nsim)) outpar = cdf(nsim)
      end if
c
c Open input file with all of the realizations and output file:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) stop 'ERROR datafl does not exist!'
      open(lin,file=datafl,status='UNKNOWN')
      open(lout,file=outfl,status='UNKNOWN')

      if(iout.eq.1.or.iout.eq.5) then
            write(lout,101)
 101        format('Conditional Variance')
            write(lout,201) 2,nx,ny,nz
            write(lout,102)
 102        format('E-type',/,'variance')
      end if
      if(iout.eq.2) then
            write(lout,103) outpar
 103        format('Probability and mean value > ',f12.4)
            write(lout,201) 3,nx,ny,nz
            write(lout,104)
 104        format('prob > cutoff',/,'mean > cutoff',/,'mean < cutoff')
      end if
      if(iout.eq.3) then
            write(lout,105) outpar
 105        format('Z value corresponding to CDF = ',f7.4)
            write(lout,201) 1,nx,ny,nz
            write(lout,106) outpar
 106        format('value',g14.4)
      end if
      if(iout.eq.4) then
            write(lout,107) outpar
 107        format('Probability Interval = ',f7.4)
            write(lout,201) 2,nx,ny,nz
            write(lout,108)
 108        format('lower',/,'upper')
      end if
      if(iout.eq.6) then
            write(lout,109) outpar
 109        format('Probability to be within ',f7.4,' percent of mean')
            write(lout,201) 2,nx,ny,nz
            write(lout,110) outpar
 110        format('E-type',/,'probability to be in ',f7.4,' percent')
      end if
 201  format(4(1x,i4))
c
c If we are getting the Z value for a particular probability we can
c establish the weighting straight away:
c
      if(iout.eq.3) then
            call locate(cdf,nsim,1,nsim,outpar,iii)
            wtiii  = (outpar-cdf(iii))/(cdf(iii+1)-cdf(iii))
            wtiiii = 1.0 - wtiii
            if(iii.eq.nsim) then
                  iii    = nsim - 1
                  wtiii  = 0.
                  wriiii = 1.
            end if      
      endif
      if(iout.eq.4) then
            outlow = (1.0-outpar)/2.0
            call locate(cdf,nsim,1,nsim,outlow,iii)
            wtiii  = (outlow-cdf(iii))/(cdf(iii+1)-cdf(iii))
            wtiiii = 1.0 - wtiii
            if(iii.eq.nsim) then
                  iii    = nsim - 1
                  wtiii  = 0.
                  wriiii = 1.
            end if      
            outupp = (1.0+outpar)/2.0
            call locate(cdf,nsim,1,nsim,outupp,jjj)
            wtjjj  = (outupp-cdf(jjj))/(cdf(jjj+1)-cdf(jjj))
            wtjjjj = 1.0 - wtjjj
            if(jjj.eq.nsim) then
                  jjj    = nsim - 1
                  wtjjj  = 0.
                  wrjjjj = 1.
            end if      
      endif
c
c MAIN LOOP OVER ALL OF THE Z LEVELS:
c
      do 2 iznow=1,nz
c
c Rewind data file and read in the distributions
c
      rewind(lin)
      read(lin,*)
      read(lin,*) nvari
      do i=1,nvari
            read(lin,*)
      end do
      do is=1,nsim
            do iz=1,nz
                  do iy=1,ny
                        do ix=1,nx
                              if(iz.eq.iznow) then
                                    read(lin,*)var(ix,iy,is)
                              else
                                    read(lin,*)
                              endif
                        end do
                  end do
            end do
      end do
c
c Now, we have the distributions to work with. Go through each node:
c
      do iy=1,ny
      do ix=1,nx
c
c Load the cut array and sort:
c
            do i=1,nsim
                  cut(i) = var(ix,iy,i)
            end do
            call sortem(1,nsim,cut,0,b,c,d,e,f,g,h)
c
c Compute the E-type?
c
            if(iout.eq.1.or.iout.eq.5) then
                  if(cut(nsim).lt.tmin) then
                        cvar  = UNEST
                  else
                        etype = 0.0
                        cvar  = 0.0
                        do is=1,nsim
                              etype = etype + cut(is)
                              cvar  = cvar  + cut(is)*cut(is)
                        end do
                        etype = etype / real(nsim)
                        cvar  = cvar  / real(nsim)-etype*etype
                  endif
                  write(lout,200) etype,cvar
c
c Compute the probability and mean above cutoff?
c
            else if(iout.eq.2) then
                  if(cut(nsim).lt.tmin) then
                        prob  = UNEST
                        meana = UNEST
                        meanb = UNEST
                  else
                        prob  = 0.0
                        meana = 0.0
                        meanb = 0.0
                        do i=1,nsim
                              if(cut(i).ge.outpar) then
                                    prob  = prob  + 1.0
                                    meana = meana + cut(i)
                              else
                                    meanb = meanb + cut(i)
                              endif
                        end do
                        if(prob.eq.0) then
                              meana = UNEST
                        else
                              meana = meana / prob
                        endif
                        if((real(nsim)-prob).eq.0) then
                              meanb = UNEST
                        else
                              meanb = meanb / (real(nsim)-prob)
                        endif
                        prob  = prob / real(nsim)
                  endif
                  write(lout,200) prob,meana,meanb
c
c Now look for the right Z value?
c
            else if(iout.eq.3) then
                  if(cut(nsim).lt.tmin) then
                        zval = UNEST
                  else
                        zval = wtiii*cut(iii)+wtiiii*cut(iii+1)
                  endif
                  write(lout,200) zval
c
c Now look for the right Z values for probability interval:
c
            else if(iout.eq.4) then
                  if(cut(nsim).lt.tmin) then
                        zlow = UNEST
                  else
                        zlow = wtiii*cut(iii)+wtiiii*cut(iii+1)
                  endif
                  if(cut(nsim).lt.tmin) then
                        zupp = UNEST
                  else
                        zupp = wtjjj*cut(jjj)+wtjjjj*cut(jjj+1)
                  endif
                  write(lout,200) zlow,zupp
c
c Compute the probability to be within a percentage of the mean?
c
            else if(iout.eq.6) then
                  if(cut(nsim).lt.tmin) then
                        etype   = UNEST
                        pwithin = UNEST
                  else
                        etype = 0.0
                        do is=1,nsim
                              etype = etype + cut(is)
                        end do
                        etype = etype / real(nsim)
                        xlo   = etype - outpar/100.0*etype
                        xhi   = etype + outpar/100.0*etype
                        pwithin = 0.0
                        do is=1,nsim
                              if(cut(is).ge.xlo.and.cut(is).le.xhi) 
     +                        pwithin = pwithin + 1.
                        end do
                        pwithin = pwithin / real(nsim)
                  endif
                  write(lout,200) etype,pwithin
            endif
c
c End loop over this ix, iy location:
c
        end do
        end do
 200  format(4(g14.8,x))
c
c End loop over this level and then loop over all levels:
c
 2      continue
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' POSTSIM Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in parameter file!'
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
      open(lun,file='postsim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for POSTSIM',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('sgsim.out                        ',
     +       '-file with simulated realizations')
      write(lun,12)
 12   format('50                               ',
     +       '-   number of realizations')
      write(lun,13)
 13   format('-0.001   1.0e21                  ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('20   20   1                      ',
     +       '-nx, ny, nz')
      write(lun,15)
 15   format('postsim.out                      ',
     +       '-file for output array(s)')
      write(lun,16)
 16   format('2   0.25                         ',
     +       '-output option, output parameter')
      write(lun,17)
 17   format(//,'option 1 = E-type mean',/,
     +          '       2 = prob and mean above threshold (par)',/,
     +          '       3 = Z-percentile corresponding to (par)',/,
     +          '       4 = symmetric (par) probability interval',/,
     +          '       5 = conditional variance')

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
      subroutine locate(xx,n,is,ie,x,j)
c-----------------------------------------------------------------------
c
c Given an array "xx" of length "n", and given a value "x", this routine
c returns a value "j" such that "x" is between xx(j) and xx(j+1).  xx
c must be monotonic, either increasing or decreasing.  j=is-1 or j=ie is
c returned to indicate that x is out of range.
c
c Bisection Concept From "Numerical Recipes", Press et. al. 1986  pp 90.
c-----------------------------------------------------------------------
      dimension xx(n)
c
c Initialize lower and upper methods:
c
      if(is.le.0) is = 1
      jl = is-1
      ju = ie
      if(xx(n).le.x) then
            j = ie
            return
      end if
c
c If we are not done then compute a midpoint:
c
 10   if(ju-jl.gt.1) then
            jm = (ju+jl)/2
c
c Replace the lower or upper limit with the midpoint:
c
            if((xx(ie).gt.xx(is)).eqv.(x.gt.xx(jm))) then
                  jl = jm
            else
                  ju = jm
            endif
            go to 10
      endif
c
c Return with the array index:
c
      j = jl
      return
      end
      subroutine sortem(ib,ie,a,iperm,b,c,d,e,f,g,h)
c-----------------------------------------------------------------------
c
c                      Quickersort Subroutine
c                      **********************
c
c This is a subroutine for sorting a real array in ascending order. This
c is a Fortran translation of algorithm 271, quickersort, by R.S. Scowen
c in collected algorithms of the ACM.
c
c The method used is that of continually splitting the array into parts
c such that all elements of one part are less than all elements of the
c other, with a third part in the middle consisting of one element.  An
c element with value t is chosen arbitrarily (here we choose the middle
c element). i and j give the lower and upper limits of the segment being
c split.  After the split a value q will have been found such that 
c a(q)=t and a(l)<=t<=a(m) for all i<=l<q<m<=j.  The program then
c performs operations on the two segments (i,q-1) and (q+1,j) as follows
c The smaller segment is split and the position of the larger segment is
c stored in the lt and ut arrays.  If the segment to be split contains
c two or fewer elements, it is sorted and another segment is obtained
c from the lt and ut arrays.  When no more segments remain, the array
c is completely sorted.
c
c
c INPUT PARAMETERS:
c
c   ib,ie        start and end index of the array to be sorteda
c   a            array, a portion of which has to be sorted.
c   iperm        0 no other array is permuted.
c                1 array b is permuted according to array a
c                2 arrays b,c are permuted.
c                3 arrays b,c,d are permuted.
c                4 arrays b,c,d,e are permuted.
c                5 arrays b,c,d,e,f are permuted.
c                6 arrays b,c,d,e,f,g are permuted.
c                7 arrays b,c,d,e,f,g,h are permuted.
c               >7 no other array is permuted.
c
c   b,c,d,e,f,g,h  arrays to be permuted according to array a.
c
c OUTPUT PARAMETERS:
c
c    a      = the array, a portion of which has been sorted.
c
c    b,c,d,e,f,g,h  =arrays permuted according to array a (see iperm)
c
c NO EXTERNAL ROUTINES REQUIRED:
c
c-----------------------------------------------------------------------
      dimension a(*),b(*),c(*),d(*),e(*),f(*),g(*),h(*)
c
c The dimensions for lt and ut have to be at least log (base 2) n
c
      integer   lt(64),ut(64),i,j,k,m,p,q
c
c Initialize:
c
      j     = ie
      m     = 1
      i     = ib
      iring = iperm+1
      if (iperm.gt.7) iring=1
c
c If this segment has more than two elements  we split it
c
 10   if (j-i-1) 100,90,15
c
c p is the position of an arbitrary element in the segment we choose the
c middle element. Under certain circumstances it may be advantageous
c to choose p at random.
c
 15   p    = (j+i)/2
      ta   = a(p)
      a(p) = a(i)
      go to (21,19,18,17,16,161,162,163),iring
 163     th   = h(p)
         h(p) = h(i)
 162     tg   = g(p)
         g(p) = g(i)
 161     tf   = f(p)
         f(p) = f(i)
 16      te   = e(p)
         e(p) = e(i)
 17      td   = d(p)
         d(p) = d(i)
 18      tc   = c(p)
         c(p) = c(i)
 19      tb   = b(p)
         b(p) = b(i)
 21   continue
c
c Start at the beginning of the segment, search for k such that a(k)>t
c
      q = j
      k = i
 20   k = k+1
      if(k.gt.q)     go to 60
      if(a(k).le.ta) go to 20
c
c Such an element has now been found now search for a q such that a(q)<t
c starting at the end of the segment.
c
 30   continue
      if(a(q).lt.ta) go to 40
      q = q-1
      if(q.gt.k)     go to 30
      go to 50
c
c a(q) has now been found. we interchange a(q) and a(k)
c
 40   xa   = a(k)
      a(k) = a(q)
      a(q) = xa
      go to (45,44,43,42,41,411,412,413),iring
 413     xh   = h(k)
         h(k) = h(q)
         h(q) = xh
 412     xg   = g(k)
         g(k) = g(q)
         g(q) = xg
 411     xf   = f(k)
         f(k) = f(q)
         f(q) = xf
 41      xe   = e(k)
         e(k) = e(q)
         e(q) = xe
 42      xd   = d(k)
         d(k) = d(q)
         d(q) = xd
 43      xc   = c(k)
         c(k) = c(q)
         c(q) = xc
 44      xb   = b(k)
         b(k) = b(q)
         b(q) = xb
 45   continue
c
c Update q and search for another pair to interchange:
c
      q = q-1
      go to 20
 50   q = k-1
 60   continue
c
c The upwards search has now met the downwards search:
c
      a(i)=a(q)
      a(q)=ta
      go to (65,64,63,62,61,611,612,613),iring
 613     h(i) = h(q)
         h(q) = th
 612     g(i) = g(q)
         g(q) = tg
 611     f(i) = f(q)
         f(q) = tf
 61      e(i) = e(q)
         e(q) = te
 62      d(i) = d(q)
         d(q) = td
 63      c(i) = c(q)
         c(q) = tc
 64      b(i) = b(q)
         b(q) = tb
 65   continue
c
c The segment is now divided in three parts: (i,q-1),(q),(q+1,j)
c store the position of the largest segment in lt and ut
c
      if (2*q.le.i+j) go to 70
      lt(m) = i
      ut(m) = q-1
      i = q+1
      go to 80
 70   lt(m) = q+1
      ut(m) = j
      j = q-1
c
c Update m and split the new smaller segment
c
 80   m = m+1
      go to 10
c
c We arrive here if the segment has  two elements we test to see if
c the segment is properly ordered if not, we perform an interchange
c
 90   continue
      if (a(i).le.a(j)) go to 100
      xa=a(i)
      a(i)=a(j)
      a(j)=xa
      go to (95,94,93,92,91,911,912,913),iring
 913     xh   = h(i)
         h(i) = h(j)
         h(j) = xh
 912     xg   = g(i)
         g(i) = g(j)
         g(j) = xg
 911     xf   = f(i)
         f(i) = f(j)
         f(j) = xf
   91    xe   = e(i)
         e(i) = e(j)
         e(j) = xe
   92    xd   = d(i)
         d(i) = d(j)
         d(j) = xd
   93    xc   = c(i)
         c(i) = c(j)
         c(j) = xc
   94    xb   = b(i)
         b(i) = b(j)
         b(j) = xb
   95 continue
c
c If lt and ut contain more segments to be sorted repeat process:
c
 100  m = m-1
      if (m.le.0) go to 110
      i = lt(m)
      j = ut(m)
      go to 10
 110  continue
      return
      end
