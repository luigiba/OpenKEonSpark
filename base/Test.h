#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

/*=====================================================================================
link prediction
======================================================================================*/
extern "C"
void getHeadBatch(INT index, INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = i;
        pt[i] = testList[index].t;
        pr[i] = testList[index].r;
    }
}

extern "C"
void getTailBatch(INT index, INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = testList[index].h;
        pt[i] = i;
        pr[i] = testList[index].r;
    }
}

extern "C"
INT* testHead(INT index, REAL *con) {
    INT h = testList[index].h;
    INT t = testList[index].t;
    INT r = testList[index].r;
    REAL minimal = con[h];

    INT* l_arr = new INT[4];
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    INT lef = 0;
    INT rig = 0;
    lef = head_lef[r];
    rig = head_rig[r];

    for (INT j = 0; j < entityTotal; j++) {
        if (j != h) {

            REAL value = con[j];
            if (value < minimal) {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }

            //TYPE_C
            while (lef < rig && head_type[lef] < j) lef ++;
            if (lef < rig && j == head_type[lef]) {
                if (value < minimal) {
                    l_s_constrain += 1;
                    if (not _find(j, t, r)) {
                        l_filter_s_constrain += 1;
                    }
                }
            }
        }
    }

    l_arr[0] = l_s;
    l_arr[1] = l_filter_s;
    l_arr[2] = l_s_constrain;
    l_arr[3] = l_filter_s_constrain;

    return l_arr;

}



extern "C"
INT* testTail(INT index, REAL *con) {
    INT h = testList[index].h;
    INT t = testList[index].t;
    INT r = testList[index].r;
    REAL minimal = con[t];

    INT* r_arr = new INT[4];
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;

    INT lef = 0;
    INT rig = 0;
    lef = tail_lef[r];
    rig = tail_rig[r];

    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {

            REAL value = con[j];
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }

            //TYPE_C
            while (lef < rig && tail_type[lef] < j) lef ++;
            if (lef < rig && j == tail_type[lef]) {
                    if (value < minimal) {
                        r_s_constrain += 1;
                        if (not _find(h, j ,r)) {
                            r_filter_s_constrain += 1;
                        }
                    }
            }


        }
    }

    r_arr[0] = r_s;
    r_arr[1] = r_filter_s;
    r_arr[2] = r_s_constrain;
    r_arr[3] = r_filter_s_constrain;

    return r_arr;
}



/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList;
extern "C"
void getNegTest() {
    negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        negTestList[i].t = corrupt(testList[i].h, testList[i].r);
    }
}

Triple *negValidList;
extern "C"
void getNegValid() {
    negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
    for (INT i = 0; i < validTotal; i++) {
        negValidList[i] = validList[i];
        negValidList[i].t = corrupt(validList[i].h, validList[i].r);
    }   
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}

extern "C"
void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}

REAL threshEntire;
extern "C"
void getBestThreshold(REAL *relThresh, REAL *score_pos, REAL *score_neg) {
    REAL interval = 0.01;
    REAL min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        total = (validRig[r] - validLef[r] + 1) * 2;
        min_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] < min_score) min_score = score_neg[validLef[r]];
        max_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] > max_score) max_score = score_neg[validLef[r]];
        for (INT i = validLef[r]+1; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        n_interval = INT((max_score - min_score)/interval);
        for (INT i = 0; i <= n_interval; i++) {
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] <= tmpThresh) correct ++;
                if (score_neg[j] > tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
            }
        }
        relThresh[r] = bestThresh;
    }
}

REAL *testAcc;
REAL aveAcc;
extern "C"
//EDIT
void test_triple_classification(REAL *relThresh, REAL *score_pos, REAL *score_neg, REAL *acc_addr) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    REAL aveAcc;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]) correct++;
            if (score_neg[i] > relThresh[r]) correct++;
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct; 
        aveTotal += total;
    }
    //EDIT
    aveAcc = 1.0 * aveCorrect / aveTotal;
    printf("triple classification accuracy is %lf\n", aveAcc);
    acc_addr[0] = 1.0 * aveCorrect / aveTotal;
}



#endif
