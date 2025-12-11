#include <bits/stdc++.h>
using namespace std;

//helpers
double sq(double x) { return x * x; }
double rmse_vec(const vector<double>& a, const vector<double>& b) {
    double s = 0;
    for (int i = 0; i < a.size(); i++) s += sq(a[i] - b[i]);
    return sqrt(s / a.size());
}

// Simple z-score scaler
struct Scaler {
    double mean = 0, stdv = 1;
    void fit(const vector<double>& v) {
        mean = accumulate(v.begin(), v.end(), 0.0) / v.size();
        double s = 0;
        for (double x : v) s += sq(x - mean);
        s /= v.size();
        stdv = (s <= 0) ? 1.0 : sqrt(s);
    }
    double transform(double x) const { return (x - mean) / stdv; }
    vector<double> transformVec(const vector<double>& v) const {
        vector<double> out; out.reserve(v.size());
        for (double x : v) out.push_back(transform(x));
        return out;
    }
};

bool invertMatrix(vector<vector<double>> A, vector<vector<double>>& inv) {
    int n = A.size();
    inv.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) inv[i][i] = 1.0;
    const double EPS = 1e-14;
    for (int i = 0; i < n; i++) {
        double pivot = A[i][i]; int pivRow = i;
        if (fabs(pivot) < EPS) {
            for (int r = i + 1; r < n; r++)
                if (fabs(A[r][i]) > fabs(pivot)) { pivot = A[r][i]; pivRow = r; }
            if (fabs(pivot) < EPS) return false;
            swap(A[i], A[pivRow]); swap(inv[i], inv[pivRow]);
        }
        pivot = A[i][i];
        for (int j = 0; j < n; j++) { A[i][j] /= pivot; inv[i][j] /= pivot; }
        for (int r = 0; r < n; r++) {
            if (r == i) continue;
            double f = A[r][i];
            for (int c = 0; c < n; c++) { A[r][c] -= f * A[i][c]; inv[r][c] -= f * inv[i][c]; }
        }
    }
    return true;
}
vector<double> matVec(const vector<vector<double>>& M, const vector<double>& v) {
    int n = M.size(); vector<double> out(n, 0.0);
    for (int i = 0; i < n; i++) for (int j = 0; j < v.size(); j++) out[i] += M[i][j] * v[j];
    return out;
}

// polynomial regression
void buildNormalEq(const vector<double>& xs, const vector<double>& ys, int degree,
                   vector<vector<double>>& XT_X, vector<double>& XT_y) {
    int D = degree + 1; XT_X.assign(D, vector<double>(D, 0.0)); XT_y.assign(D, 0.0);
    int N = xs.size();
    vector<vector<double>> powx(N, vector<double>(2*D+1,1.0));
    for (int i = 0; i < N; i++) for (int p = 1; p < 2*D+1; p++) powx[i][p] = powx[i][p-1]*xs[i];
    for (int r = 0; r < D; r++) {
        for (int c = 0; c < D; c++) { double s = 0; for (int i = 0; i < N; i++) s += powx[i][r+c]; XT_X[r][c]=s; }
        double sy=0; for (int i=0;i<N;i++) sy+=powx[i][r]*ys[i]; XT_y[r]=sy;
    }
}

bool fit_poly(const vector<double>& xs, const vector<double>& ys, int degree, double ridge, vector<double>& coef_out) {
    vector<vector<double>> XT_X; vector<double> XT_y;
    buildNormalEq(xs, ys, degree, XT_X, XT_y);
    for (int i = 0; i <= degree; i++) XT_X[i][i] += ridge;
    vector<vector<double>> inv;
    if(!invertMatrix(XT_X, inv)) return false;
    coef_out = matVec(inv, XT_y);
    return true;
}
vector<double> predict_poly(const vector<double>& xs, const vector<double>& coef) {
    vector<double> out; out.reserve(xs.size());
    for(double x: xs) { double xi=1,sum=0; for(double c: coef){ sum+=c*xi; xi*=x;} out.push_back(sum); }
    return out;
}

// logistic regression
struct Logistic {
    double L;      // upper asymptote
    double k;      // growth rate
    double x0;     // midpoint
    double b;      // vertical shift
	double output_scale = 1.0;

    Logistic() : L(1), k(1), x0(0), b(0) {}

    double predict(double x) const {
        return L / (1 + exp(-k * (x - x0))) + b;
    }

	void fit_gd_stable(const vector<double>& X, const vector<double>& Y,
                   int iters = 5000, double lr = 0.01)
{
    // Scale Y to [0,1] for numerical stability
    double Ymin = *min_element(Y.begin(), Y.end());
    double Ymax = *max_element(Y.begin(), Y.end());
    double scale = (Ymax - Ymin) > 1e-9 ? (Ymax - Ymin) : 1.0;

    vector<double> Ys_scaled;
    Ys_scaled.reserve(Y.size());
    for(double y : Y) Ys_scaled.push_back((y - Ymin)/scale);

    // Init parameters
L = *max_element(Y.begin(), Y.end());
b = *min_element(Y.begin(), Y.end());

    x0 = X[X.size()/2];
    k = 0.1;

    for(int it=0; it<iters; it++){
        double dL=0, dk=0, dx0=0, db=0;

        for(int j=0; j<X.size(); j++){
            double x = X[j], y = Ys_scaled[j];
            double z = -k*(x - x0);
            // clamp z to avoid overflow
            if(z < -50) z = -50;
            if(z > 50) z = 50;
            double ex = exp(z);
            double pred = L / (1 + ex) + b;
            double e = pred - y;
            double base = ex / ((1 + ex)*(1 + ex));

            dL  += e / (1 + ex);
            dk  += e * L * (x - x0) * base;
            dx0 += e * L * k * base;
            db  += e;
        }

        // small updates for stability
        L  -= lr * dL;
        k  -= lr * dk;
        x0 -= lr * dx0;
        b  -= lr * db;

        // clamp k to reasonable range
        if(k > 5) k = 5;
        if(k < -5) k = -5;
    }

    // scale back L and b to original Y range
    L = L * scale;
    b = b * scale + Ymin;
}

};


// CV for poly & logistic
vector<double> timeseries_cv_poly(const vector<double>& xs, const vector<double>& ys, int degree, int folds, double ridge) {
    int n=xs.size(); vector<double> rmses;
    int val_size=max(2,n/(folds+1));
    for(int f=1;f<=folds;f++){
        int train_end=f*val_size; int val_start=train_end; int val_end=min(n,val_start+val_size);
        if(val_start>=n) break;
        vector<double> xtr(xs.begin(),xs.begin()+train_end), ytr(ys.begin(),ys.begin()+train_end);
        vector<double> xv(xs.begin()+val_start,xs.begin()+val_end), yv(ys.begin()+val_start,ys.begin()+val_end);
        vector<double> coef; bool ok=fit_poly(xtr,ytr,degree,ridge,coef);
        if(!ok){ rmses.push_back(1e18); continue; }
        auto pred=predict_poly(xv,coef);
        rmses.push_back(rmse_vec(pred,yv));
    }
    return rmses;
}

vector<double> timeseries_cv_logistic(const vector<double>& xs, const vector<double>& ys, int folds){
    int n=xs.size(); vector<double> rmses; if(folds<2) return rmses;
    int val_size=max(2,n/(folds+1));
    double y_max=*max_element(ys.begin(),ys.end());
    vector<double> ys_scaled; for(double y: ys) ys_scaled.push_back(y/y_max);
    for(int f=1;f<=folds;f++){
        int train_end=f*val_size; int val_start=train_end; int val_end=min(n,val_start+val_size);
        if(val_start>=n) break;
        vector<double> xtr(xs.begin(),xs.begin()+train_end);
        vector<double> ytr(ys_scaled.begin(),ys_scaled.begin()+train_end);
        vector<double> xv(xs.begin()+val_start,xs.begin()+val_end);
        vector<double> yv(ys.begin() + val_start, ys.begin() + val_end);
	Logistic L;  L.fit_gd_stable(xtr, ytr, 5000, 0.01);
        vector<double> pred;
	pred.reserve(xv.size());
	for(double xx : xv) pred.push_back(L.predict(xx) * y_max);
	rmses.push_back(rmse_vec(pred,vector<double>(yv.begin(),yv.end())));
    }
    return rmses;
}
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    ifstream fin("data.txt"); if(!fin){ cerr<<"data.txt missing\n"; return 1; }
    vector<double> x,y; double a,b; while(fin>>a>>b){ x.push_back(a); y.push_back(b); }
    if(x.size()<8){ cerr<<"need 8+ rows\n"; return 1; }

    const int MAX_DEGREE=2, CV_FOLDS=5; const double RIDGE=1e-3; const int HORIZON=10;

    vector<pair<int,double>> deg_rmse;
    for(int d=1;d<=MAX_DEGREE;d++){
        auto rmses=timeseries_cv_poly(x,y,d,CV_FOLDS,RIDGE);
        double mean=accumulate(rmses.begin(),rmses.end(),0.0)/rmses.size();
        deg_rmse.push_back({d,mean});
        cout<<"Degree "<<d<<" CV RMSE = "<<mean<<"\n";
    }

    bool logistic_ok=all_of(y.begin(),y.end(),[](double v){return v>0;});
    double logistic_rmse=1e18;
    if(logistic_ok){
        auto lrmses=timeseries_cv_logistic(x,y,CV_FOLDS);
        logistic_rmse=accumulate(lrmses.begin(),lrmses.end(),0.0)/lrmses.size();
        cout<<"Logistic CV RMSE = "<<logistic_rmse<<"\n";
    } else cout<<"Logistic skipped (non-positive)\n";

    int best_deg=deg_rmse[0].first; double best_deg_rmse=deg_rmse[0].second;
    for(auto &p:deg_rmse) if(p.second<best_deg_rmse){ best_deg=p.first; best_deg_rmse=p.second; }
    cout<<"\nBest polynomial degree (CV): "<<best_deg<<" RMSE="<<best_deg_rmse<<"\n";

    vector<double> final_coef; fit_poly(x,y,best_deg,RIDGE,final_coef);
    Logistic final_log; if(logistic_ok){
        double y_max=*max_element(y.begin(),y.end());
        vector<double> y_scaled; for(double v:y) y_scaled.push_back(v/y_max);
	final_log.fit_gd_stable(x, y_scaled, 5000, 0.01);
	final_log.output_scale = y_max;
    }

    double w_poly=1.0, w_log=0.0;
    if(logistic_ok){
        double inv_poly=1.0/(best_deg_rmse+1e-9), inv_log=1.0/(logistic_rmse+1e-9);
        double sum=inv_poly+inv_log; w_poly=inv_poly/sum; w_log=inv_log/sum;
    }

    double poly_cv = best_deg_rmse;
	double log_cv  = logistic_rmse;


if (log_cv < poly_cv * 0.8) {
    w_poly = 0.0;
    w_log  = 1.0;
}

    cout<<"\nEnsemble weights: poly="<<w_poly<<" logistic="<<w_log<<"\n";

    cout<<"\nPredictions next "<<HORIZON<<" x steps:\n";
    double last_x=x.back();
    for(int i=1;i<=HORIZON;i++){
        double xi=last_x+i;
        double polyp=0.0,powx=1.0; for(double c:final_coef){ polyp+=c*powx; powx*=xi; }
        double logp = logistic_ok ? final_log.predict(xi) : 0.0;
	double en=w_poly*polyp + w_log*logp;
        cout<<xi<<" : "<<en<<" (poly="<<polyp<<", log="<<logp<<")\n";
    }

    return 0;
}

