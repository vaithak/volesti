#include "ecos.h"
#include "minunit.h"
idxint random_infeasible_n = 120;
idxint random_infeasible_m = 120;
idxint random_infeasible_p = 12;
idxint random_infeasible_l = 0;
idxint random_infeasible_ncones = 20;
idxint random_infeasible_nexc = 20;
pfloat random_infeasible_c[120] = {3.138609313386915289e-01, 1.050766883873559099e+00, 2.219878700931303683e+00, -3.647275223554702794e+00, -2.380727542044204892e+00, 2.887405970236490926e+00, -4.193715639979892273e+00, 7.569896482257483239e-01, -1.333906938498313322e+00, -2.148909962490122005e+00, 3.010261062624862483e+00, 6.487515684685138950e-01, -1.091543757217773525e+00, -9.645061823532143919e-01, -2.747048450764828509e+00, -4.699929266154031282e+00, -3.019459124109577619e+00, -1.808896872609794260e+00, 3.766656297880055515e-01, 6.873578077377263673e-01, 3.883087783467722431e-01, -1.149598723066540318e+00, -1.047951081334355106e-01, -1.587104533535094453e-01, -1.406621392438331952e+00, -1.270894926291498361e+00, -1.892785603897124691e+00, -1.088862219111182750e+00, -2.798883420615674300e-01, 5.051069289268323992e-01, -2.742510552292422688e+00, -2.764286483137761152e+00, -7.915732851848646945e-01, -1.180202518425532476e+00, -2.914283633447828370e-02, -5.730921639893202801e-01, -1.473755622110032970e+00, 1.292707540827759960e+00, 4.008269980634970864e-03, -1.147494890540345480e+00, -8.050330389471840320e-01, -3.525489588647210404e+00, -3.721297300450744139e+00, -1.303216049191175396e-01, 2.643992695567614071e-01, -5.061010271527565330e+00, -6.320708080276465912e-01, -4.104083758764918155e-01, -3.372604249738439197e+00, 1.735743153764724100e+00, 6.704785725193507240e-02, -6.205783487083548877e+00, 1.299585012866592670e+00, -2.164376839429353439e+00, -2.347128968378302094e+00, -1.654540030549568996e+00, -8.131737567297633307e-01, -1.892276770707047806e+00, -8.137594745404810315e-01, -9.920565671134482510e-01, -2.272123163176055449e+00, -4.306496201116064571e-01, 1.484872735836117119e+00, 1.966794229547274675e+00, 9.524443282769905483e-01, -7.039012150590662209e-03, 2.732182045123590974e+00, -1.306610978525364786e+00, -8.335664719762834185e-01, -4.971768322927695993e-01, 5.786849577333863515e-02, -1.883149889963248613e+00, 2.221937874465523821e-01, -1.892637374470247380e-01, -1.104572357399676186e+00, -9.962359063749153254e-01, -2.090035921249017825e+00, 2.638214342871560980e-01, -2.687952829457807002e-01, -2.429748116037024452e+00, -1.921526643978950233e-01, -7.955414600536305780e-01, 1.402044758152793946e-01, 1.100018828416760019e-02, 1.554366710410208796e+00, -4.324783673906857917e-01, -1.826366143845188361e+00, 1.388869981175955282e-01, -2.567686613514834626e-01, -1.370548359997666887e+00, -5.754456291407425983e-01, -2.927704309932646520e+00, -3.239768128216220955e+00, 1.337634871749739718e+00, 3.247628789094589852e-01, 1.131782982094454049e+00, 1.288138765171471078e+00, 6.636348669519089860e-01, -1.900861456567876973e+00, -8.990311034582936234e-01, -8.836156763504594558e-01, -3.936387601060460373e-01, 1.501006490338027577e+00, -7.322531727246042177e-01, 6.489937198014450459e-01, -4.028765639626261663e-01, -1.506883214497029666e+00, 1.544859252331759247e+00, -1.354102988623022474e+00, 7.620466887048931159e-01, -1.446225240198625972e+00, 1.419520286133117981e-01, -4.341583463106240437e+00, -8.708159754836763966e-02, 2.777927285483582587e+00, -8.708266093280594422e-01, -5.371679306029870649e-01, -1.081601045445151454e+00, 2.905199369635262929e-01, -1.348046627108409501e+00};
idxint random_infeasible_Gjc[121] = {0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
idxint random_infeasible_Gir[120] = {0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
pfloat random_infeasible_Gpr[120] = {1.000000000000000222e+00, 9.999999999999997780e-01, 1.000000000000000000e+00, 1.000000000000000444e+00, 1.000000000000000444e+00, 1.000000000000000666e+00, 1.000000000000000222e+00, 1.000000000000000666e+00, 1.000000000000000222e+00, 1.000000000000000444e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000888e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000666e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 9.999999999999997780e-01, 9.999999999999998890e-01, 1.000000000000000444e+00, 9.999999999999998890e-01, 1.000000000000000222e+00, 9.999999999999998890e-01, 1.000000000000000666e+00, 1.000000000000000222e+00, 1.000000000000000444e+00, 1.000000000000000222e+00, 9.999999999999998890e-01, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 9.999999999999996669e-01, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000444e+00, 9.999999999999998890e-01, 9.999999999999997780e-01, 1.000000000000000666e+00, 9.999999999999997780e-01, 1.000000000000000444e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000222e+00, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 9.999999999999998890e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00};
pfloat random_infeasible_h[120] = {6.064719434570648993e-01, -1.109593658017167916e+00, -5.166674516853461130e-01, -1.836976196268667039e-01, -3.594783806369653556e-01, 1.385422149933343006e+00, -5.806240719137419903e-01, 5.614773265625411236e-02, -6.287355633251562326e-01, 5.445266322095648759e-01, 8.528959284622239859e-03, -7.902609128102382474e-01, -3.099641592169860571e-01, 1.295791219279868844e+00, -1.019425244594354352e+00, -1.268260053482529459e+00, -5.509230839995108431e-01, 3.320676817511180601e-01, 1.855038668596811213e+00, 2.438724311381212040e+00, -1.347960700710906901e+00, -8.969716378248651090e-01, -9.947074802043344155e-01, 2.103064294824744440e+00, 2.727030416489178988e-02, -2.976851107773658134e-01, -3.351857218063025723e-01, 6.691805980495204587e-01, -3.559793869412926637e-01, 2.879879931408796945e-01, -3.915907942050925183e-01, 8.427227285979921323e-01, -1.254286992228576647e+00, 4.337934984906108293e-02, 4.987238795534357783e-01, 1.888301183502217206e+00, -3.688929146819007188e-01, 1.026195111361016543e-01, 1.861330562749874185e+00, -5.233444240203425496e-02, 1.527279089759913777e+00, 1.557180270473959238e+00, -1.311125401318528338e-01, -1.583171152169096374e-01, 1.233582035834257740e+00, -4.703411994041042932e-01, 6.766787730939309853e-01, -8.000992826008215264e-01, 1.096479587584225257e+00, -7.361433461276409296e-02, 1.445933952672672662e+00, -3.382300060116523455e-02, -1.389732343413651629e-02, -1.196673705835893525e-01, -1.131730284654351770e+00, -1.589903113045942895e+00, -2.590938658563429753e-01, -1.105002105643759780e+00, -3.918473793179677034e-01, 1.368649400488590739e+00, 8.633912843887228128e-01, -9.991194182927988221e-01, 2.014527461367568717e-01, 3.768108158580632527e-01, -1.238002319787778749e+00, -1.805197167847779893e-01, -6.678326558403417934e-01, 4.167484747934523392e-01, -8.064649758703650306e-01, -1.082275497746127829e-01, -1.026262590412440501e+00, -3.595045542500092117e-01, 1.560955903242181864e-01, 4.524936943363312980e-01, 6.979676321872946461e-03, 1.127271734785422197e+00, 8.167675300952599304e-01, 1.736989925444317961e+00, -4.411633490846739725e-01, -2.226032901794653962e-01, 9.205866680686818349e-01, 3.779455456067871543e-03, 3.604277753680475843e-02, 4.380959858077315694e-01, 5.728035256698772448e-01, 1.290977072998285569e-01, -1.273967422192194610e+00, 4.604419959197799533e-02, 1.207045745473133147e+00, 4.643636493666356002e-02, -5.195161148658472561e-01, 1.661138346913955122e-01, 4.513828059163264572e-01, 1.386638180250965968e+00, -5.527051359807133579e-01, 4.591316460125732801e-02, -6.436690064723255000e-01, -6.966619023360057228e-02, -6.802325971391899317e-01, -7.139533984367310571e-02, 6.069422255573146963e-01, 1.148101709772105378e-01, 5.833260510086650807e-01, 1.603542459239243545e+00, 3.114875048392829471e-01, 6.688633453561733111e-01, 1.547050541637780441e+00, -3.703906116477885258e-01, -4.951148211470010607e-01, -4.847432153278817268e-01, -1.752961479549453028e-02, 1.987973823191321043e-01, 2.239023355890074410e-02, 4.585058497245639297e-01, -3.645851730582876926e-01, -1.332965540880531208e+00, -7.215924003887150828e-01, -5.525138325644769477e-01, -9.067911626514831847e-01, -2.954326682627993472e-01};
idxint random_infeasible_q[20] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
idxint random_infeasible_Ajc[121] = {0.0, 6, 9, 15, 18, 21, 25, 29, 33, 36, 39, 42, 51, 55, 60, 67, 73, 79, 84, 88, 97, 102, 106, 110, 116, 121, 124, 129, 136, 141, 145, 150, 156, 163, 166, 172, 175, 180, 186, 191, 194, 197, 204, 212, 213, 218, 224, 226, 235, 239, 245, 251, 260, 265, 271, 277, 283, 287, 294, 299, 307, 311, 316, 323, 328, 335, 341, 349, 352, 357, 362, 370, 374, 380, 384, 389, 394, 400, 407, 413, 418, 427, 434, 441, 443, 450, 456, 460, 465, 469, 475, 481, 488, 494, 503, 506, 512, 519, 526, 532, 538, 543, 547, 553, 560, 564, 567, 572, 578, 581, 586, 591, 594, 601, 607, 614, 620, 626, 630, 635, 640};
idxint random_infeasible_Air[640] = {0.0, 1, 3, 4, 8, 11, 3, 6, 11, 2, 3, 6, 7, 8, 11, 0.0, 4, 11, 0.0, 7, 11, 1, 7, 8, 11, 2, 5, 6, 11, 2, 3, 6, 11, 5, 10, 11, 2, 5, 11, 3, 5, 11, 1, 3, 4, 5, 6, 8, 9, 10, 11, 3, 4, 10, 11, 0.0, 3, 5, 9, 11, 0.0, 2, 3, 4, 5, 7, 11, 0.0, 1, 2, 5, 9, 11, 2, 4, 6, 7, 9, 11, 2, 4, 7, 10, 11, 3, 5, 9, 11, 0.0, 1, 2, 3, 4, 5, 7, 10, 11, 1, 3, 6, 9, 11, 0.0, 1, 5, 11, 0.0, 6, 10, 11, 2, 5, 6, 7, 8, 11, 2, 3, 6, 9, 11, 0.0, 9, 11, 2, 5, 7, 8, 11, 0.0, 1, 4, 5, 9, 10, 11, 3, 4, 7, 10, 11, 1, 4, 6, 11, 0.0, 3, 9, 10, 11, 0.0, 3, 5, 6, 8, 11, 0.0, 1, 2, 3, 6, 9, 11, 7, 10, 11, 4, 6, 7, 8, 9, 11, 5, 9, 11, 3, 4, 7, 8, 11, 1, 3, 6, 7, 8, 11, 0.0, 3, 7, 10, 11, 3, 7, 11, 0.0, 2, 11, 2, 5, 6, 7, 9, 10, 11, 0.0, 1, 2, 5, 7, 9, 10, 11, 11, 0.0, 6, 8, 10, 11, 0.0, 2, 5, 6, 7, 11, 9, 11, 0.0, 2, 4, 5, 6, 7, 9, 10, 11, 0.0, 3, 5, 11, 1, 2, 3, 4, 7, 11, 0.0, 1, 3, 5, 8, 11, 0.0, 2, 4, 5, 6, 7, 9, 10, 11, 1, 3, 4, 6, 11, 1, 2, 7, 9, 10, 11, 0.0, 2, 3, 7, 9, 11, 0.0, 1, 5, 7, 9, 11, 3, 6, 8, 11, 0.0, 1, 2, 4, 6, 7, 11, 3, 5, 7, 10, 11, 1, 2, 4, 5, 6, 7, 10, 11, 0.0, 2, 9, 11, 4, 7, 8, 10, 11, 0.0, 1, 2, 3, 5, 6, 11, 3, 4, 9, 10, 11, 0.0, 1, 2, 3, 4, 9, 11, 1, 2, 4, 8, 10, 11, 1, 3, 5, 6, 8, 9, 10, 11, 7, 8, 11, 0.0, 2, 3, 8, 11, 5, 8, 9, 10, 11, 1, 3, 4, 6, 8, 9, 10, 11, 2, 3, 9, 11, 2, 4, 6, 9, 10, 11, 2, 3, 6, 11, 0.0, 3, 5, 8, 11, 1, 2, 5, 7, 11, 2, 4, 8, 9, 10, 11, 1, 4, 5, 6, 7, 8, 11, 4, 5, 6, 7, 9, 11, 2, 6, 8, 10, 11, 0.0, 1, 2, 3, 4, 5, 6, 9, 11, 5, 6, 7, 8, 9, 10, 11, 0.0, 3, 4, 7, 8, 9, 11, 4, 11, 1, 3, 5, 6, 7, 8, 11, 0.0, 1, 3, 5, 7, 11, 2, 3, 10, 11, 2, 3, 7, 10, 11, 1, 2, 5, 11, 1, 4, 5, 9, 10, 11, 3, 5, 6, 7, 10, 11, 0.0, 4, 5, 7, 8, 9, 11, 0.0, 1, 5, 7, 9, 11, 0.0, 1, 3, 4, 6, 8, 9, 10, 11, 1, 10, 11, 0.0, 1, 3, 6, 7, 11, 1, 2, 3, 6, 7, 10, 11, 0.0, 1, 2, 3, 6, 8, 11, 0.0, 2, 3, 6, 10, 11, 0.0, 1, 2, 8, 10, 11, 1, 2, 8, 10, 11, 4, 5, 10, 11, 0.0, 3, 4, 6, 10, 11, 1, 5, 6, 7, 9, 10, 11, 0.0, 3, 7, 11, 7, 10, 11, 4, 7, 9, 10, 11, 1, 3, 5, 6, 8, 11, 2, 7, 11, 1, 6, 8, 9, 11, 0.0, 8, 9, 10, 11, 0.0, 9, 11, 2, 3, 5, 7, 9, 10, 11, 4, 5, 6, 8, 10, 11, 0.0, 1, 3, 7, 8, 9, 11, 3, 4, 6, 9, 10, 11, 2, 3, 7, 8, 9, 11, 2, 6, 9, 11, 1, 2, 4, 6, 11, 3, 7, 9, 10, 11};
pfloat random_infeasible_Apr[640] = {9.564574030232725343e-01, 8.390399999375705020e-01, 9.385254910041894716e-01, 7.302354632279581814e-01, 9.111485343188482355e-01, 2.431242268207024271e+00, 3.040648462537880459e-01, 9.917877888890971327e-01, 1.404185263514527549e+00, 4.476762172541876028e-01, 8.458064943726182339e-01, 8.381195298186262521e-01, 4.842056991285207390e-01, 5.648171797140546513e-01, -1.948010874771419099e+00, 8.360035623513456526e-01, 5.630448261764868789e-01, 2.582285419201961307e+00, 3.980235337022924313e-01, 7.832305250859422641e-01, 3.481680641005593380e-01, 7.239910434132047490e-01, 1.607211097124379706e-01, 5.714515257648148383e-01, -2.806496670745822097e+00, 9.803485825109364837e-01, 8.308093249599525221e-01, 4.881677528933039967e-01, 3.012678187651230299e+00, 8.458990479882916302e-01, 5.083693131154983780e-01, 9.162114015187815541e-01, -2.160084091268574369e+00, 4.855476882956286655e-01, 8.829721606930733824e-01, 2.429356313452236682e-01, 3.562979682679645599e-02, 5.645238959948724045e-01, 3.943079872349986914e+00, 9.823613543480927968e-01, 2.445525948866492660e-01, -8.526442372474886400e-02, 7.619333518368932578e-01, 3.317472476224098132e-01, 2.998936318161105063e-02, 9.245380313924765892e-01, 1.994298570222236000e-01, 7.366800552480976538e-01, 5.106643027160869819e-01, 9.207942740923806824e-01, 1.325951035660605681e+00, 4.181757205332312100e-01, 8.728445154291445407e-01, 9.633851936007840067e-01, 2.509927579463650638e+00, 7.816264897564433323e-01, 8.582661849700918832e-01, 4.674286694993311508e-01, 1.434776671399965176e-01, 1.893861046885698229e+00, 4.715239574334587069e-01, 4.237683796080936038e-01, 4.840099984088220286e-01, 4.384012343461913264e-01, 6.450580191789205831e-01, 8.711910989464527466e-01, 1.839837420874044449e+00, 3.263215529322304742e-01, 3.230871116533950982e-01, 9.580369975281920469e-01, 9.300734257129850135e-01, 1.630510312173644727e-01, 2.507623698179691107e+00, 8.090430561847646862e-01, 7.152374248454608230e-01, 4.869188853628255709e-01, 7.649278958616899660e-01, 1.144383180949043316e-01, 1.001309811618419365e+00, 2.995100425927692545e-01, 6.630850377088511050e-02, 4.940417388633089324e-01, 8.638818845297582261e-01, -2.046123897607895259e+00, 9.918680871054058601e-01, 3.478953609351177478e-01, 4.526295875158595772e-01, 4.054905196957698799e+00, 2.383827295980176786e-01, 7.191202725187654243e-01, 2.259808750212660422e-01, 6.112842696258788955e-01, 9.422184282921449716e-01, 3.562361307629793306e-01, 6.003547407822948934e-01, 8.973500967950021057e-01, 6.096518112693978531e-02, 2.125663041458091540e-01, 3.095385140363236531e-01, 6.012178521664106601e-01, 5.352916148842006283e-01, 2.235035102619374214e+00, 1.682926935996644779e-01, 2.951198912719732448e-01, 7.582109586347401953e-01, 2.899078774088704513e+00, 1.404611228748562923e-01, 5.574642725849162650e-01, 4.895069403136082520e-01, -9.752684798601044536e-01, 1.785999303701339969e-01, 1.329401039739962764e-01, 8.729494708200815634e-01, 6.329724370622026586e-02, 1.142427517603543108e-01, 7.693738120380400058e-01, 3.871549499699495356e-01, 7.156894920142601313e-01, 7.422119402436873070e-01, 8.200665789076719836e-01, 4.116800590878234800e+00, 5.324497516050973323e-01, 8.434644856133025614e-01, -1.064236460246499982e+00, 2.454711771574310808e-01, 5.573787415574116277e-01, 2.157350988358013311e-01, 3.498300117472802340e-01, 1.811029373101779472e+00, 7.125448445358559431e-03, 3.161258771314184313e-01, 7.834680021684412576e-01, 3.718955769805684097e-01, 3.025699904460548706e-01, 2.310208299842673452e-01, 3.898184484722415188e+00, 1.617057421260625072e-01, 9.506372344121108986e-01, 1.677207934298330649e-01, 3.744073334266870279e-01, 1.629459811941395442e+00, 1.939106230088122051e-01, 3.733330746211032114e-01, 3.251432557627385034e-01, 1.096670647371700058e+00, 7.176648893889933656e-02, 2.099119160589594801e-01, 8.676201478697530556e-01, 1.848028666523867380e-01, 3.989938696763938974e+00, 8.520292220769354330e-01, 1.630105951054481306e-01, 4.490525469723954388e-01, 4.353253005374481399e-02, 4.865500216533337752e-01, 1.211738155212127355e+00, 6.030303108919592425e-01, 2.698630005967063150e-01, 4.979124899257769687e-01, 3.198467452114795440e-01, 3.544065423645978430e-01, 5.062501255252476184e-01, -1.988057003265398581e+00, 4.821887599288864762e-02, 3.334504668143445927e-01, 8.271712650533376010e-01, 5.709704277371224768e-01, 7.598195643886418438e-01, 5.411955036422257947e-01, 1.136607175544532000e-01, 5.731390128912090542e-01, 1.221186327265983751e+00, 4.666083423282087472e-01, 1.338667171827222779e-01, 1.472749659234040998e+00, 6.243558198384041580e-01, 1.938602641068543075e-02, 9.491285071166634113e-01, 7.351284526792863927e-01, 3.130052535514041612e+00, 6.205055383804132518e-02, 6.229256180389714653e-02, 9.667781021119156160e-01, 2.913925109923369283e-01, 5.443913658326705596e-01, -6.366881514786476570e-01, 3.957758362869402968e-01, 2.049092380002099922e-01, 4.784649319005739865e-01, 5.225883282467590485e-01, -3.445846354961001001e+00, 6.509326702545078502e-01, 2.517639816447526102e-01, 4.236103643271472485e+00, 7.270661378821500209e-01, 4.073140240063793382e-01, -4.741006593545546544e+00, 7.948635105828233405e-01, 1.798368164139070025e-01, 7.711768818159896455e-01, 3.201803436981505713e-01, 9.274908997339085248e-01, 7.889941808821780977e-01, -5.732759860757210557e-01, 3.210241938305002235e-02, 4.206103944241779691e-01, 5.635783779438621188e-01, 3.525062388497041677e-01, 6.132976480549050535e-01, 7.065730184571799785e-03, 6.031652878049417188e-01, 1.500755465435569258e+00, 2.054441677787316456e-01, 5.383980344972421062e-01, 2.857291425436735421e-01, 1.293693519122568747e-01, 6.555982116744506205e-02, -2.902783646950650276e+00, 5.327010276777714992e-01, 6.362266899520730989e-01, 8.324094821713019510e-01, 3.625810813767671448e-01, 7.364122151543741612e-01, 2.108047409827707330e+00, 2.002071566836071681e-01, 9.114381986470081465e-01, 7.500543265050528541e-02, 1.297259742527328286e-01, 4.852136617933805462e-01, 1.310457062634174004e-02, 7.354587386396048698e-01, 9.913762766577195462e-01, 1.738156148305238236e-01, 6.774376170215080650e-01, -1.987226694472201682e+00, 9.850314139091354004e-01, 2.363204525028220204e-01, 6.505489416044170081e-01, 2.218867906989057115e+00, 8.477480311702461080e-01, 4.228342609505404415e-01, 1.679219256388292392e-01, 1.990036775187204732e-01, 1.983061115189288862e-02, -1.384937376686826926e+00, 3.166535924950773101e-02, 3.511703053039403333e-01, 2.611131341532773043e-01, 9.824661663866858241e-01, 5.892661487165017087e-02, 3.279603306923516648e+00, 6.720007793773099536e-01, 4.123570041553688559e-01, 3.645475371158910516e-01, 2.687661658125964359e-01, 2.250745681845110757e-01, 9.068997208512776798e-01, 9.736754110351056868e-01, 5.279915628601272032e-02, 3.318770006556322816e+00, 9.253700627103405019e-01, 2.624686355924266690e-01, 3.641887453662707719e-01, 9.062660373448456763e-01, 4.286413685419703690e+00, 6.685140984151645149e-01, 2.344532942868579883e-01, 7.341836162435619872e-01, 6.147702254612014139e-01, 2.567849700920377121e-01, 1.895346357199267695e+00, 4.086032039955470041e-01, 3.350204753185862527e-01, 7.464931033870583210e-01, 1.688492751393791125e-01, 4.852897939299776753e-01, 2.509279654627822875e+00, 3.555104483355777112e-01, 5.651174858912215715e-01, 6.687796544560340450e-01, 9.424714538470104763e-01, 5.467364505611380654e-01, 1.577511389046131196e+00, 1.245788334199288833e-01, 1.668123138686062809e-02, 1.048237876094083171e-01, 1.815658904624440684e+00, 2.570386996733011742e-01, 1.708426625555364842e-01, 6.066669254867679273e-01, 5.374387953638463111e-01, 7.807027102751898928e-01, 2.381558447895220643e-01, 7.423654017537547611e-01, 4.564431908681516048e-01, 9.984163878347751542e-02, 9.359907224383310353e-01, 9.891534676898606770e-01, -1.032115979861393251e+00, 4.899615504169120839e-01, 5.001332048824942422e-01, 7.673975951270736173e-01, 7.742123204878195164e-01, 6.280274081307577472e-01, 3.404187170376572480e-01, 8.078030608429435100e-01, 9.770527530817051387e-01, 2.936462927542734525e-01, 8.188146854641987771e-01, 6.979863301296160838e-01, -2.852504278283256145e+00, 8.893823049208153808e-01, 2.468248887242447676e-01, 6.883040997217956813e-02, 2.492681888878438279e-01, 1.223124532888707927e+00, 2.749377704482946494e-01, 9.452714100351311544e-01, 2.378720223621074992e-01, 5.901872666212173435e-01, 7.198568386287741185e-03, 4.760335469889424576e-01, 1.314208599722106463e+00, 8.424076982128132496e-01, 2.308557363141716320e-02, 1.280409793349635839e-01, 4.078434460593570732e-01, -5.554393010088654048e-01, 2.477262138582471129e-01, 6.806046141104203917e-01, 3.615084680351037805e-01, 8.664304467204932969e-01, 5.873160789183744512e-01, 3.590375687889664769e-01, 2.125183669717339185e+00, 8.679818598727979673e-01, 4.611698871790444465e-01, 1.630839701968822900e-02, 3.602376180842729037e-01, 1.788202564343424894e-01, -5.115585844595098575e-01, 9.738041853113982294e-01, 7.171084366255373599e-01, 6.126414196565845849e-01, 9.772127874227993626e-04, 4.434299131425685214e-01, 2.845788460943531328e-01, 7.093647032491122717e-01, 4.070647343265461071e-01, 6.185670471805591042e-01, 3.308099823238975778e-01, 8.298276258835582952e-01, 3.038106610930064733e-01, 7.418731632454207414e-01, 8.469630313301501623e-01, 1.493833388984497079e-01, -7.679142757117767282e-01, 2.760498756490935568e-01, 8.743109752921275346e-01, 8.864323007813665889e-01, 8.658883186956480849e-01, -2.056589192306593539e+00, 4.324004368225377593e-01, 1.384916592438943661e-01, 4.648122197954418500e-02, 7.351480593099894711e-02, 5.744661715219718801e-01, 1.614946800006341610e-01, 3.464323191401127433e-01, 7.951266341954710271e-01, 6.640168230441247932e-01, 1.342041444418510809e-01, 4.109972282456458847e-01, -3.797626089625415480e-01, 1.053674945317465417e-01, 3.809928097180146911e-01, 4.585632602987254614e-01, 7.013677559794896377e-01, 2.959239321026639891e-01, -5.564984215309680682e-01, 9.910866979968059598e-02, 2.634357052546540401e-01, 5.922003084734606126e-01, 1.924891273211576825e+00, 7.439850825927649680e-01, 2.571458015602777203e-01, 4.081631853675851129e-01, 9.948571272266358845e-01, -7.861673498438765950e-01, 7.860354403850600136e-01, 6.608134156215150767e-01, 9.839499780518164984e-01, 9.249670342385857280e-01, -1.978325441519457328e-02, 6.810343505818184928e-01, 6.783578263743994530e-01, 2.667594574843039634e-01, 3.910311723053050970e-01, 2.209485343992846351e-01, -1.655916778866770578e-01, 5.978939605677486835e-01, 5.182546438223007979e-01, 8.639874970339556093e-01, 4.172541027877736397e-01, 6.368742598263820165e-01, 2.295786398996487176e-01, 3.124124333712244628e+00, 9.341810332317617149e-01, 9.349921796556431852e-01, 7.023899681154458507e-01, 9.791820924799004011e-01, 8.328956694111018244e-01, 2.622033419172592250e+00, 6.673393578512015756e-01, 6.946031723537421376e-01, 5.055674389467434970e-01, 9.148699174422691982e-01, -1.566912153033046096e+00, 2.238233524312584766e-01, 6.544580911528455891e-01, 4.622786573951869205e-01, 1.344442636596060725e-01, 9.565030445129047987e-01, 3.025467992871246348e-01, 6.174950269877467202e-01, 8.584910038159985213e-01, 2.147260844063146212e+00, 3.989480616947653768e-01, 4.090322705807706294e-01, 8.021434247059225919e-01, 4.459182990362571819e-01, 6.976900564190382514e-01, 3.584159066129006255e-01, -4.190547812724668697e-01, 5.105163215105164021e-01, 8.635249917563871946e-01, 8.158083386506225221e-01, 4.825412900586245057e-01, 5.854038566093145990e-01, 3.139392535646755578e-01, 1.332127098577991831e+00, 4.134319418244127298e-01, 7.808846193487910536e-01, 6.130096569010823337e-02, 5.818465501146293795e-01, 5.616648141264403149e-01, 3.609300395186832433e-01, 5.866051972056509634e-01, 8.384415566516144613e-01, 2.978707763271339903e-01, 6.151549914872971314e-01, 3.562443918205170368e-01, 9.410984696387169057e-01, 1.209424578701324138e-01, 8.097180500085249255e-01, 1.555220979112634350e+00, 9.459321318359193986e-01, 4.975081691607242673e-01, 4.910591539941319694e-01, -1.688225047550692803e+00, 6.280473392186410875e-01, 7.128590650255969274e-01, 3.848224416842443985e-01, 2.254563265171392050e-01, -1.956107439624175326e+00, 4.612587665820541694e-01, 2.387221055605280931e-02, 2.578368565673534452e-01, 2.062431479169414938e+00, 3.387269205134703087e-01, 7.930580830058098218e-02, 8.547317681149223523e-01, 8.496833357381504692e-01, 6.969729642527041635e-01, 2.180226384816361662e+00, 6.159737034832081515e-02, 1.783641335848657472e-01, 1.295720838523217906e-02, 6.589477779393965173e-01, 8.094822894844555528e-01, -2.022253902825821292e+00, 2.612079603260923366e-01, 2.671578103751847798e-03, 9.461063931642357261e-01, 6.100277611513521547e-01, 3.975666860197250019e-01, 8.719551136116676648e-01, 2.947312667413651610e+00, 9.105780398249300189e-01, 3.955482818499367337e-02, 8.789459286962945095e-01, 5.470006647713485659e-01, 2.499079484415861474e-01, 7.074066495878970073e-01, 7.656589783418413231e-02, 4.007724901562446096e-01, 4.527017482112229152e-02, 5.420405471871986292e-01, 9.651897456927095442e-01, 8.003656148727379405e-01, 9.398362536963895675e-01, 5.562955767452039124e-01, -6.426793299286326366e-01, 8.444930389106457547e-01, 2.583994268936388727e-01, 1.402395687700944249e+00, 4.946448689427593859e-01, 4.564693213703297503e-01, 9.301955610092482729e-01, 1.403739207380139131e-01, 2.025430863479867338e-01, 1.119286270970653474e+00, 8.012261761731768184e-01, 1.734238814379939972e-02, 6.275621579365122174e-03, 2.198704153824452279e-01, 5.305807397560600158e-01, 2.298960378614115785e-01, -7.947802686976301878e-01, 2.904802192326394125e-02, 3.775850698580844322e-01, 5.316128789806384702e-02, 3.889974826380854789e-01, 3.137762796702662782e-01, 2.861350527241194319e-01, 1.826932459628612948e+00, 5.338851885782093243e-01, 7.689053676404891124e-01, 4.681243930514241702e-01, 8.900020603610157011e-01, 2.619436867806567704e-01, -1.218457621800568935e+00, 3.684255371377685995e-01, 5.584190417156512654e-01, 5.894739203461893950e-01, 1.408428067401657491e-01, 6.194347166602949262e-01, -3.870599244985184484e+00, 4.029266423866778024e-01, 4.155745979842836446e-01, 7.360266842340950078e-01, 8.921896298778797529e-02, -2.664532792756986179e-01, 6.114931414472462023e-01, 7.014887787438304212e-01, 3.137835306052853546e-01, 1.849991312054310999e+00, 1.779122806064518114e-01, 2.846832513620732863e-01, 8.620498702829505477e-01, 1.132661110014387962e-01, 2.758092179789058962e-01, -9.399394399619805940e-01, 7.409579733709767968e-01, 4.233260727055123684e-01, 1.600918209404763870e-01, 5.799208965144512229e-01, 1.959362261819641918e-02, 4.733282090219323446e-01, 1.948643200600137737e+00, 2.429817655493869410e-01, 8.530675917772081451e-01, 2.332483385547757804e-01, 1.091480400360854253e+00, 9.986392354256409254e-01, 6.562451500169598495e-02, -1.310214497273825884e+00, 3.561245442188711863e-01, 3.148836632046236317e-01, 1.375704815295816508e-01, 7.812671579997253479e-01, 2.280131747278923982e-01, 7.917244627473942753e-01, 4.561503679067611672e-01, 6.222597567387926354e-01, 5.248410452903552736e-02, 6.082080701249580057e-02, 2.960108542513382623e+00, 8.172507157295045843e-01, 2.026555313966415806e-01, -2.857050962910498537e+00, 6.525528587219420196e-01, 8.065732815014693413e-01, 6.997801019495180075e-01, 1.648081525471216113e-01, 1.877029123215153250e+00, 5.692598439939173804e-01, 6.000843997326262835e-01, 2.067840010632538295e-03, 2.823406502707300758e-01, -1.676172437673986826e+00, 6.624280694212092424e-02, 5.009007932395161955e-01, -7.735166491258416155e-01, 8.199109727326735486e-01, 2.021329487026094496e-01, 9.465511210438665168e-01, 2.619756104557572907e-01, 8.741516497653640805e-01, 9.352776635831347996e-01, 8.632447577252930770e-01, 8.471980435791619524e-01, 4.443954273592052306e-01, 8.201945754422866885e-01, 3.359506517974662071e-02, 9.158603862767110693e-01, 1.172284288552648102e+00, 2.017845589984118837e-01, 9.858715896283589686e-01, 8.340801591967803574e-01, 8.162833098610314808e-01, 9.319412604727834726e-01, 2.851693722082383475e-01, -5.898540986770592909e-01, 2.250470950578177011e-03, 4.520193520725987457e-01, 3.479606613844427043e-01, 2.729332219902452228e-01, 5.753714473447747446e-01, 1.106448061458926713e+00, 4.444742882661151806e-01, 8.529692110914811609e-01, 8.772284071997512589e-01, 9.749551267666128895e-01, 4.765427438921646108e-01, -7.402953193494443695e-02, 9.813820569379140268e-01, 9.971048176469591295e-01, 9.722467636266853008e-02, -2.172848258670251109e+00, 9.828116282896783851e-01, 5.644357047792012416e-01, 7.561676081389093751e-01, 2.781938392074778665e-01, 1.366570408779187673e+00, 4.690901130179799261e-01, 6.395475498969007688e-01, 6.825617389181729555e-01, 9.806840633262186024e-01, -1.163424554061287530e-02};
pfloat random_infeasible_b[12] = {1.419807807744515038e-01, -6.332827047969422640e-01, 9.394061259826161359e-02, -4.523124346045784838e-01, -1.201119450356671114e+00, -1.391724704670400659e+00, 1.433950571444463085e-01, 1.354154236259927924e-01, 2.179888648989162014e+00, -1.462052010056586804e+00, 5.197953538683719454e-02, -3.222863969661456940e-01};
 
static char * test_random_infeasible(){
pwork *mywork;
idxint exitflag;
 
/* print test name */
printf("================================== random_infeasible ==================================\n");
 
/* set up data */
mywork = ECOS_setup(random_infeasible_n, random_infeasible_m, random_infeasible_p, random_infeasible_l, random_infeasible_ncones, random_infeasible_q, random_infeasible_nexc,
                    random_infeasible_Gpr, random_infeasible_Gjc, random_infeasible_Gir,
                    random_infeasible_Apr, random_infeasible_Ajc, random_infeasible_Air,
                    random_infeasible_c, random_infeasible_h, random_infeasible_b);
if( mywork != NULL ){
/* solve */
exitflag = ECOS_solve(mywork); }
else exitflag = ECOS_FATAL;
 
/* clean up memory */
ECOS_cleanup(mywork, 0);
 
mu_assert("random_infeasible: ECOS failed to produce outputflag ECOS_PINF", exitflag == ECOS_PINF );
return 0;
}
