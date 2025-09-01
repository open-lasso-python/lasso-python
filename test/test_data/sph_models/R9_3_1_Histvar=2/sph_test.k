*KEYWORD
*TITLE
$#                                                                         title
SPH Unit Test
*CONTROL_CONTACT
$#  slsfac    rwpnal    islchk    shlthk    penopt    thkchg     orien    enmass
       0.0       0.0         1         0         1         0         1         0
$#  usrstr    usrfrc     nsbcs    interm     xpene     ssthk      ecdt   tiedprj
         0         0         0         0       0.0         0         0         0
*CONTROL_CPU
$#  cputim     iglst
       0.0
*CONTROL_DAMPING
$#  nrcyck     drtol    drfctr    drterm    tssfdr    irelal     edttl    idrflg
         0       0.0       0.0       0.0       0.0         0       0.0         0
*CONTROL_ENERGY
$#    hgen      rwen    slnten     rylen     irgen     maten     drlen     disen
         2         2         2         1         2         1         1         1
*CONTROL_OUTPUT
$#   npopt    neecho    nrefup    iaccop     opifs    ipnint    ikedit    iflush
         0         0         0         0       0.0         0       100      5000
*CONTROL_SPH
$#    ncbs     boxid        dt      idim   nmneigh      form     start      maxv
         0         0       0.0         2         0         0       0.0       0.0
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol
     0.035         0       0.0       0.0       0.0         0
*CONTROL_TIMESTEP
$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
       0.0       0.9         0       0.0       0.0         0         0         0
*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt
4.00000E-4         2         0         1
*DATABASE_MATSUM
$#      dt    binary      lcur     ioopt
4.00000E-4         2         0         1
*DATABASE_RCFORC
$#      dt    binary      lcur     ioopt
4.00000E-4         2         0         1
*DATABASE_RWFORC
$#      dt    binary      lcur     ioopt
4.00000E-4         2         0         1
*DATABASE_SLEOUT
$#      dt    binary      lcur     ioopt
4.00000E-4         2         0         1
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
     0.004         0         0         0         0
*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         2         0         0         0         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         1         1         1         2         1
*MAT_PLASTIC_KINEMATIC
$#     mid        ro         e        pr      sigy      etan      beta
         1   0.00271   68900.0      0.33     117.0       6.8       1.0
$#     src       srp        fs        vp
       0.0       0.0       0.0       0.0
*MAT_PLASTIC_KINEMATIC
$#     mid        ro         e        pr      sigy      etan      beta
         2   0.00271   68900.0      0.33     117.0       6.8       1.0
$#     src       srp        fs        vp
       0.0       0.0       0.0       0.0
*INITIAL_VELOCITY
$#    nsid    nsidex     boxid    irigid      icid
         0         0         0         0         0
$#      vx        vy        vz       vxr       vyr       vzr
     -50.0       0.0       0.0       0.0       0.0       0.0
*PART
$#                                                                         title

$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
       101       101         1         0         0         0         0         0
*SECTION_SPH
$#   secid      cslh      hmin      hmax    sphini     death     start   sphkern
       101       1.3       0.0       0.0       0.0       0.0       0.0         0
*RIGIDWALL_PLANAR_ID
$#      id                                                                 title
         1                                                                      
$#    nsid    nsidex     boxid    offset     birth     death     rwksf
         0         0         0       0.0       0.01.00000E20       1.0
$#      xt        yt        zt        xh        yh        zh      fric      wvel
       0.0       0.0       0.0       1.0       0.0       0.0       0.0       0.0
*SET_NODE_LIST_GENERATE
$#     sid       da1       da2       da3       da4    solver       its         -
       101       0.0       0.0       0.0       0.0MECH               1          
$#   b1beg     b1end     b2beg     b2end     b3beg     b3end     b4beg     b4end
   1000001   1004800         0         0         0         0         0         0
*NODE
$#   nid               x               y               z      tc      rc
 1001801       0.4998333       -11.04725             0.0       0       0
 1001802          0.7115       -11.04725             0.0       0       0
 1001803       0.9231666       -11.04725             0.0       0       0
 1001804        1.134833       -11.04725             0.0       0       0
 1001805          1.3465       -11.04725             0.0       0       0
 1001921       0.4998333       -10.85675             0.0       0       0
 1001922          0.7115       -10.85675             0.0       0       0
 1001923       0.9231666       -10.85675             0.0       0       0
 1001924        1.134833       -10.85675             0.0       0       0
 1001925          1.3465       -10.85675             0.0       0       0
 1002041       0.4998333       -10.66625             0.0       0       0
 1002042          0.7115       -10.66625             0.0       0       0
 1002043       0.9231666       -10.66625             0.0       0       0
 1002044        1.134833       -10.66625             0.0       0       0
 1002045          1.3465       -10.66625             0.0       0       0
 1002161       0.4998333       -10.47575             0.0       0       0
 1002162          0.7115       -10.47575             0.0       0       0
 1002163       0.9231666       -10.47575             0.0       0       0
 1002164        1.134833       -10.47575             0.0       0       0
 1002165          1.3465       -10.47575             0.0       0       0
 1002281       0.4998333       -10.28525             0.0       0       0
 1002282          0.7115       -10.28525             0.0       0       0
 1002283       0.9231666       -10.28525             0.0       0       0
 1002284        1.134833       -10.28525             0.0       0       0
 1002285          1.3465       -10.28525             0.0       0       0
*ELEMENT_SPH
$#   nid     pid            mass      nend
 1001801     101    1.092736e-04         0
 1001802     101    1.092736e-04         0
 1001803     101    1.092736e-04         0
 1001804     101    1.092736e-04         0
 1001805     101    1.092735e-04         0
 1001921     101    1.092741e-04         0
 1001922     101    1.092741e-04         0
 1001923     101    1.092741e-04         0
 1001924     101    1.092741e-04         0
 1001925     101    1.092741e-04         0
 1002041     101    1.092741e-04         0
 1002042     101    1.092741e-04         0
 1002043     101    1.092741e-04         0
 1002044     101    1.092741e-04         0
 1002045     101    1.092741e-04         0
 1002161     101    1.092741e-04         0
 1002162     101    1.092741e-04         0
 1002163     101    1.092741e-04         0
 1002164     101    1.092741e-04         0
 1002165     101    1.092741e-04         0
 1002281     101    1.092736e-04         0
 1002282     101    1.092736e-04         0
 1002283     101    1.092736e-04         0
 1002284     101    1.092736e-04         0
 1002285     101    1.092735e-04         0
*END
