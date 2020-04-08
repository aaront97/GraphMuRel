#include <bits/stdc++.h>
using namespace std;

#define Fi first
#define Se second
typedef long long ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> pll;
#define all(x) (x).begin(), (x).end()
#define pb push_back
#define szz(x) (int)(x).size()
#define rep(i, n) for(int i=0;i<(n);i++)
typedef tuple<int, int, int> t3;

int B;

int Query(int x) {
	printf("%d\n", x);
	fflush(stdout);
	int a; scanf("%d", &a);
	return a;
}

void solve() {
	int ans[110] = {};
	memset(ans, -1, sizeof ans);
	auto Rev = [&]() { reverse(ans+1, ans+1+B); };
	auto Flip = [&]() { for(int i=1;i<=B;i++) if(ans[i] != -1) ans[i] = 1 - ans[i]; };
	for(int i=1;i<=5;i++) {
		ans[i] = Query(i);
		ans[B+1-i] = Query(B+1-i);
	}
	int s = 5;
	while(s * 2 < B) {
		int cq = 0;
		int f1 = -1, f2 = -1;
		for(int i=1;i<=s;i++) {
			if(ans[i] != ans[B+1-i]) f1 = i;
			else f2 = i;
		}
		if(f2 != -1) {
			int v1 = Query(f2); ++cq;
			if(v1 != ans[f2]) Flip();
		}
		if(f1 != -1) {
			int v1 = Query(f1); ++cq;
			if(v1 != ans[f1]) Rev();
		}
		int t = min((10 - cq) / 2, B - s / 2);
		for(int j=1;j<=t;j++) {
			ans[s + j] = Query(s + j);
			ans[B+1-s - j] = Query(B+1-s - j);
			cq += 2;
		}
		while(cq < 10) Query(1), cq++;
		s += t;
	}
	for(int i=1;i<=B;i++) printf("%d", ans[i]); puts("");

	fflush(stdout);
	char ch[4];
	scanf("%s", ch);
	if(ch[0] == 'N') exit(0);
}

int main() {
	int T; scanf("%d%d", &T, &B);
	for(int t=1;t<=T;t++) {
		solve();
	}
	return 0;
}
