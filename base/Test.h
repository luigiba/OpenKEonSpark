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

    INT* l_arr = new INT[8];
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    INT l_min = h;
    INT l_filter_min = h;
    INT l_constrain_min = h;
    INT l_filter_constrain_min = h;

    REAL l_min_s = minimal;
    REAL l_filter_min_s = minimal;
    REAL l_constrain_min_s = minimal;
    REAL l_filter_constrain_min_s = minimal;

    INT lef = 0;
    INT rig = 0;
    lef = head_lef[r];
    rig = head_rig[r];

    for (INT j = 0; j < entityTotal; j++) {
        if (j != h) {

            REAL value = con[j];
            if (value < minimal) {
                l_s += 1;
                if (value < l_min_s){
                    l_min_s = value;
                    l_min = j;
                }

                if (not _find(j, t, r))
                    l_filter_s += 1;
                    if (value < l_filter_min_s){
                        l_filter_min_s = value;
                        l_filter_min = j;
                    }
            }

            //TYPE_C
            while (lef < rig && head_type[lef] < j) lef ++;
            if (lef < rig && j == head_type[lef]) {
                if (value < minimal) {
                    l_s_constrain += 1;
                    if (value < l_constrain_min_s){
                        l_constrain_min_s = value;
                        l_constrain_min = j;
                    }

                    if (not _find(j, t, r)) {
                        l_filter_s_constrain += 1;
                        if (value < l_filter_constrain_min_s){
                            l_filter_constrain_min_s = value;
                            l_filter_constrain_min = j;
                        }
                    }
                }
            }
        }
    }

    l_arr[0] = l_s;
    l_arr[1] = l_filter_s;
    l_arr[2] = l_s_constrain;
    l_arr[3] = l_filter_s_constrain;

    l_arr[4] = l_min;
    l_arr[5] = l_filter_min;
    l_arr[6] = l_constrain_min;
    l_arr[7] = l_filter_constrain_min;

    INT lef_sup = sup_lef[h];
    INT rig_sup = sup_rig[h];
    INT lef_sub = sub_lef[h];
    INT rig_sub = sub_rig[h];
    for(INT i = 4; i < 8; i++){
        if (l_arr[i] == h){                                                         //it's ok
            l_arr[i] = 0;
            continue;
        }

        while (lef_sup < rig_sup && sup_type[lef_sup] < l_arr[i]) lef_sup ++;       //generalization error
        if (lef_sup < rig_sup && l_arr[i] == sup_type[lef_sup]){
            l_arr[i] = 1;
            continue;
        }

        while (lef_sub < rig_sub && sub_type[lef_sub] < l_arr[i]) lef_sub ++;       //specialization error
        if (lef_sub < rig_sub && l_arr[i] == sub_type[lef_sub]){
            l_arr[i] = 2;
            continue;
        }

        l_arr[i] = 3;                                                               //misclassification error
    }

    return l_arr;

}



extern "C"
INT* testTail(INT index, REAL *con) {
    INT h = testList[index].h;
    INT t = testList[index].t;
    INT r = testList[index].r;
    REAL minimal = con[t];

    INT* r_arr = new INT[8];

    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;

    INT r_min = t;
    INT r_filter_min = t;
    INT r_constrain_min = t;
    INT r_filter_constrain_min = t;

    REAL r_min_s = minimal;
    REAL r_filter_min_s = minimal;
    REAL r_constrain_min_s = minimal;
    REAL r_filter_constrain_min_s = minimal;


    INT lef = tail_lef[r];
    INT rig = tail_rig[r];
    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {

            REAL value = con[j];
            if (value < minimal) {
                r_s += 1;
                if (value < r_min_s){
                    r_min_s = value;
                    r_min = j;
                }

                if (not _find(h, j, r)){
                    r_filter_s += 1;
                    if (value < r_filter_min_s){
                        r_filter_min_s = value;
                        r_filter_min = j;
                    }
                }

            }

            //TYPE_C
            while (lef < rig && tail_type[lef] < j) lef ++;
            if (lef < rig && j == tail_type[lef]) {
                    if (value < minimal) {
                        r_s_constrain += 1;
                        if (value < r_constrain_min_s){
                            r_constrain_min_s = value;
                            r_constrain_min = j;
                        }


                        if (not _find(h, j ,r)) {
                            r_filter_s_constrain += 1;
                            if (value < r_filter_constrain_min_s){
                                r_filter_constrain_min_s = value;
                                r_filter_constrain_min = j;
                            }
                        }
                    }
            }


        }
    }

    r_arr[0] = r_s;
    r_arr[1] = r_filter_s;
    r_arr[2] = r_s_constrain;
    r_arr[3] = r_filter_s_constrain;

    r_arr[4] = r_min;
    r_arr[5] = r_filter_min;
    r_arr[6] = r_constrain_min;
    r_arr[7] = r_filter_constrain_min;

    INT lef_sup = sup_lef[t];
    INT rig_sup = sup_rig[t];
    INT lef_sub = sub_lef[t];
    INT rig_sub = sub_rig[t];
    for(INT i = 4; i < 8; i++){
        if (r_arr[i] == t){                                                         //it's ok
            r_arr[i] = 0;
            continue;
        }

        while (lef_sup < rig_sup && sup_type[lef_sup] < r_arr[i]) lef_sup ++;       //generalization error
        if (lef_sup < rig_sup && r_arr[i] == sup_type[lef_sup]){
            r_arr[i] = 1;
            continue;
        }

        while (lef_sub < rig_sub && sub_type[lef_sub] < r_arr[i]) lef_sub ++;       //specialization error
        if (lef_sub < rig_sub && r_arr[i] == sub_type[lef_sub]){
            r_arr[i] = 2;
            continue;
        }

        r_arr[i] = 3;                                                               //misclassification error
    }

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
    REAL accuracy, precision, recall, fmeasure;
    INT TP = 0, TN = 0, FP = 0, FN = 0;

    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]){
                correct++;
                TP++;
            }
            else{
                FN++;
            }

            if (score_neg[i] > relThresh[r]){
                correct++;
                TN++;
            }
            else{
                FP++;
            }
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
    }

    accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN);
    precision = 1.0 * TP / (TP + FP);
    recall = 1.0 * TP / (TP + FN);
    fmeasure = (2 * precision * recall) / (precision + recall);

    printf("triple classification accuracy is %lf\n", accuracy);
    printf("triple classification precision is %lf\n", precision);
    printf("triple classification recall is %lf\n", recall);
    printf("triple classification f-measure is %lf\n", fmeasure);

    acc_addr[0] = 1.0 * (TP + TN) / (TP + TN + FP + FN);
}


extern "C"
INT get_n_interval(INT r, REAL *score_pos, REAL *score_neg){
    REAL min_score, max_score;
    INT n_interval, total;
    if (validLef[r] == -1) return 0;
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
    return INT((max_score - min_score)/interval);
}


extern "C"
INT* get_TPFP(INT r, REAL *score_pos, REAL *score_neg, REAL *score_pos_test, REAL *score_neg_test) {
    REAL min_score, max_score, tmpThresh;
    INT n_interval, total;


    if (validLef[r] == -1) return 0;
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


    INT* TPFPs = new INT[(n_interval+1)*2];
    for (INT i = 0; i <= n_interval; i++) {
        INT TP = 0, FP = 0;
        tmpThresh = min_score + i * interval;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos_test[i] <= tmpThresh) TP++;
            if (score_neg_test[i] <= tmpThresh) FP++;
        }
        TPFPs[i] = TP;
        TPFPs[i + n_interval+1] = FP;
    }

    return TPFPs;
}



#endif
