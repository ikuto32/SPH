// WindowsProject_optimization_SPH.cpp : アプリケーションのエントリ ポイントを定義します。
//

#include "framework.h"
#include "WindowsProject_optimization_SPH.h"
#include <random>

#define MAX_LOADSTRING 100
#define _ENABLE_ATOMIC_ALIGNMENT_FIX

thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

float floatRand()
{
        return dist01(rng);
}

//グリッドを用いて当たり判定の対象を絞るクラス
class GridMap
{
private:
	//基本
	int width;
	int height;
	float chunkRange;

	//チャンク単位の範囲
	int numChunkWidth;
	int numChunkHeight;

	//チャンク数
	int numChunk;

	//チャンクの配列(チャンクインデックス, チャンク内の要素数)
	std::vector<std::vector<int>> chunks;

public:
	GridMap(float width, float height, float radius)
	{
		//基本
		this->width = width;
		this->height = height;
		this->chunkRange = radius;

		//チャンク単位の横幅、縦幅を求める。(端に対応するチャンクがないため +1 する)
		this->numChunkWidth = (int)std::ceil(this->width / (float)this->chunkRange) + 1;
		this->numChunkHeight = (int)std::ceil(this->height / (float)this->chunkRange) + 1;

		//チャンクの個数
		this->numChunk = this->numChunkWidth * this->numChunkHeight;

		std::vector<std::vector<int>> chunks(numChunk, std::vector<int>(0, 0));
		this->chunks = chunks;
	}

	/** チャンクの座標からチャンクに登録された対象のリストを取得する */
	std::vector<int> getChunk(int chunkX, int chunkY)
	{
		return this->chunks[chunkY * this->numChunkWidth + chunkX];
	}


	/** 対象を対応するチャンクに登録する */
	void registerTarget(int target, float x, float y)
	{
		int chunkX = (int)(x / this->chunkRange);
		int chunkY = (int)(y / this->chunkRange);
		if (chunkX < 0 || chunkX >= this->numChunkWidth)
		{
			return;
		};

		if (chunkY < 0 || chunkY >= this->numChunkHeight)
		{
			return;
		};
		this->chunks[chunkY * this->numChunkWidth + chunkX].push_back(target);
	}

	/** チャンクに登録された対象を削除する */
	void unregisterAll()
	{
		for (int i = 0; i < this->numChunk; i++)
		{
			this->chunks[i].clear();
		}
	}

	//近傍を探す
	std::vector<int> findNeighborhood(float x, float y, float radius)
	{
		std::vector<int> out;

		int centerChunkX = (int)(x / this->chunkRange);
		int centerChunkY = (int)(y / this->chunkRange);

		int radiusChunk = (int)std::ceil(radius / this->chunkRange);

		int minChunkX = centerChunkX - radiusChunk;
		int maxChunkX = centerChunkX + radiusChunk;
		int minChunkY = centerChunkY - radiusChunk;
		int maxChunkY = centerChunkY + radiusChunk;

		for (int x1 = minChunkX; x1 <= maxChunkX; x1++)
		{
			for (int y1 = minChunkY; y1 <= maxChunkY; y1++)
			{
				//範囲外の場合は無視
				if (x1 < 0 || x1 >= this->numChunkWidth) continue;
				if (y1 < 0 || y1 >= this->numChunkHeight) continue;

				std::vector<int> chunk = this->getChunk(x1, y1);
				out.insert(std::end(out), std::begin(chunk), std::end(chunk));
			}
		}

		return out;
	}
};


//力の発生源の位置と力の大きさを示す
struct  ForcePoint
{

	float pos[2];
	float radius;
	float strength;

};



class World
{


	//粒子の個数
	const static int numParticle = 1000;


	//世界の設定
	const int particleRadius = 5;
	const float gravity = 9.8F;
	const float worldSize[2] = { 20, 10 };
	const float collisionDamping = 1.0F;
	const float smoothingRadius = 0.8;
	const float targetDensity = 32.0F;
	const float pressureMultiplier = 100;
	const float delta = 0.0;
	const float drag = 0.9999;


	//粒子の各パラメータ
	float pos[numParticle][2];
	float predpos[numParticle][2];
	float vel[numParticle][2];
	float density[numParticle];
	float pressureAccelerations[numParticle][2];
	float interactionForce[numParticle][2];
	int color[numParticle][3];
	std::vector<std::vector<int>> querysize;
	std::vector<int> iterator;
	float mass[numParticle];

	ForcePoint forcePoint = { {0, 0}, 0, 0 };

	GridMap gridmap = GridMap(worldSize[0], worldSize[1], smoothingRadius);


public:
	World()
	{
		std::memset(pos, 0.0, sizeof(pos));
		std::memset(predpos, 0.0, sizeof(predpos));
		std::memset(vel, 0.0, sizeof(vel));
		std::memset(density, 0.0, sizeof(density));
		std::memset(pressureAccelerations, 0.0, sizeof(pressureAccelerations));
		std::memset(interactionForce, 0.0, sizeof(interactionForce));
		for (int i = 0; i < numParticle; i++)
		{
			mass[i] = 1;
		}


		for (int i = 0; i < numParticle; i++)
		{
			int a = std::sqrt(numParticle);
			int row = i / a;
			int col = i % a;

			pos[i][0] = (col / static_cast<float>(a)) * worldSize[0];
			pos[i][1] = (row / static_cast<float>(a)) * worldSize[1];
		}

		std::memset(color, 255, sizeof(color));

		iterator.resize(numParticle);
		for (int i = 0; i < numParticle; i++)
		{
			iterator[i] = i;
		}
	}


	void setInteractionForce(float posX, float posY, float radius, float strength)
	{
		this->forcePoint = { {posX, posY}, radius, strength };
	}

	void deleteInteractionForce()
	{
		this->forcePoint = { {0, 0}, 0, 0 };
	}

	float getWorldWidth() const
	{
		return this->worldSize[0];
	}

	float getWorldHeight() const
	{
		return this->worldSize[1];
	}


	void paint(HDC hdc, int height)
	{
		float rate = this->getWorldWidth() / this->getWorldHeight();

		this->paint(hdc, rate * height, height);
	}

	void paint(HDC hdc, int width, int height)
	{
		int scaleX = width / this->getWorldWidth();
		int scaleY = height / this->getWorldHeight();

		HBRUSH hbr = CreateSolidBrush(RGB(255, 255, 255));
		for (int i = 0; i < numParticle; i++)
		{
			hbr = CreateSolidBrush(RGB(color[i][0], color[i][1], color[i][2]));
			SelectObject(hdc, hbr);
			Ellipse(
				hdc,
				(int)(pos[i][0] * scaleX - particleRadius),
				(int)(pos[i][1] * scaleY - particleRadius),
				(int)(pos[i][0] * scaleX + particleRadius),
				(int)(pos[i][1] * scaleY + particleRadius)
			);
			DeleteObject(hbr);
		}
	}

	void update(float deltaTime)
	{
		predictedPos(deltaTime);

		gridmap.unregisterAll();

		std::vector<int> v = this->iterator;

		std::for_each(std::execution::seq, v.begin(), v.end(), [&](int particleIndex) {
			gridmap.registerTarget(particleIndex, pos[particleIndex][0], pos[particleIndex][1]);
			});

		querysize.resize(numParticle);
		std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int particleIndex) {
			querysize[particleIndex] = gridmap.findNeighborhood(pos[particleIndex][0], pos[particleIndex][1], smoothingRadius);
			});

		std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int particleIndex) {
			updateDensity(particleIndex);
			});

		std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int particleIndex) {
			updatePressureForce(particleIndex);
			});

		std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int particleIndex) {
			updateInteractionForce(particleIndex);
			});

		updatePosition(deltaTime);

		std::for_each(std::execution::par_unseq, v.begin(), v.end(), [&](int particleIndex) {
			fixPositionFromWorldSize(particleIndex);
			});

		updateColor();
	}

	void predictedPos(float deltaTime)
	{
		for (int i = 0; i < numParticle; i++)
		{
			vel[i][0] += 0.0;
			vel[i][1] += mass[i] * gravity * deltaTime;

			predpos[i][0] = pos[i][0] + vel[i][0] * 1.0 / 120.0;
			predpos[i][1] = pos[i][1] + vel[i][1] * 1.0 / 120.0;
		}
	}

	void updateDensity(int particleIndex)
	{
		density[particleIndex] = calcDensity(particleIndex);
	}

	void updatePressureForce(int particleIndex)
	{
		float pressureForce[] = { 0, 0 };
		calcPressureForce(pressureForce, particleIndex);

		//TODO 謎の処理
		//self.pressureAccelerations = np.array(pressureForces) / self.densities[:, None]

		pressureAccelerations[particleIndex][0] = pressureForce[0] / (density[particleIndex] + delta);
		pressureAccelerations[particleIndex][1] = pressureForce[1] / (density[particleIndex] + delta);

	}

	void updateInteractionForce(int i)
	{
		float outInteractionForce[] = { 0, 0 };
		calcInteractionForce(outInteractionForce, i);

		interactionForce[i][0] = outInteractionForce[0];
		interactionForce[i][1] = outInteractionForce[1];
	}

	void updatePosition(float deltaTime)
	{
		for (int i = 0; i < numParticle; i++)
		{


			vel[i][0] += (pressureAccelerations[i][0] + interactionForce[i][0]) * deltaTime;
			vel[i][1] += (pressureAccelerations[i][1] + interactionForce[i][1]) * deltaTime;

			pos[i][0] += vel[i][0] * deltaTime;
			pos[i][1] += vel[i][1] * deltaTime;

			vel[i][0] *= drag;
			vel[i][1] *= drag;
		}
	}

	void fixPositionFromWorldSize(int i)
	{
		float x = pos[i][0];
		float y = pos[i][1];
		float velX = vel[i][0];
		float velY = vel[i][1];
		int w = (int)worldSize[0];
		int h = (int)worldSize[1];

		if (x < 0)
		{
			pos[i][0] = 0;
			vel[i][0] = -velX * collisionDamping;
		}

		if (w < x)
		{
			pos[i][0] = w;
			vel[i][0] = -velX * collisionDamping;
		}

		if (y < 0)
		{
			pos[i][1] = 0;
			vel[i][1] = -velY * collisionDamping;
		}

		if (h < y)
		{
			pos[i][1] = h;
			vel[i][1] = -velY * collisionDamping;
		}
	}

	void updateColor()
	{
                float speeds[numParticle];
                float minSpeed = FLT_MAX;
                float maxSpeed = 0.0;
		int color1[3];
		color1[0] = 0;
		color1[1] = 0;
		color1[2] = 255;

		int color2[3];
		color2[0] = 255;
		color2[1] = 0;
		color2[2] = 0;
		for (int i = 0; i < numParticle; i++)
		{
			float speed = std::sqrt(vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1]);
                        if (minSpeed > speed)
                        {
                                minSpeed = speed;
                        }

                        if (maxSpeed < speed)
                        {
                                maxSpeed = speed;
                        }
			speeds[i] = speed;
		}

		for (int i = 0; i < numParticle; i++)
		{
			// calculate normalized velocity
			float normSpeed = (speeds[i] - minSpeed) / (maxSpeed - minSpeed);

			// clamp normalized velocity
			normSpeed = fminf(fmaxf(normSpeed, 0.0f), 1.0f);

			// convert normalized velocity to byte
			uint8_t byteVel = (uint8_t)(normSpeed * 255.0f);

			// set color
			color[i][0] = (255 - byteVel) * color1[0] + byteVel * color2[0];
			color[i][1] = (255 - byteVel) * color1[1] + byteVel * color2[1];
			color[i][2] = (255 - byteVel) * color1[2] + byteVel * color2[2];
		}
	}

	float calcDensity(int particleIndex)
	{
		const float radius = smoothingRadius;

		float density = 0.0F;
		std::vector<int> otherIndexes = querysize[particleIndex];
		for (int nr = 0; nr < otherIndexes.size(); nr++)
		{
			int j = otherIndexes[nr];
			float dx = predpos[j][0] - predpos[particleIndex][0];
			float dy = predpos[j][1] - predpos[particleIndex][1];
			float dist = (float)std::sqrt(dx * dx + dy * dy);

			float influence = calcSmoothingKernel(dist, radius);
			density += mass[j] * influence;
		}

		return density;
	}

	void calcPressureForce(float pressureForce[], int particleIndex)
	{
		std::vector<int> otherIndexes = querysize[particleIndex];
		for (int nr = 0; nr < otherIndexes.size(); nr++)
		{
			int otherIndex = otherIndexes[nr];
			if (particleIndex == otherIndex) continue;

			float offsetX = pos[otherIndex][0] - pos[particleIndex][0];
			float offsetY = pos[otherIndex][1] - pos[particleIndex][1];
			float dist = (float)std::sqrt(offsetX * offsetX + offsetY * offsetY);

			if (dist > smoothingRadius) continue;

			float dirX = 0;
			float dirY = 0;
			if (dist <= FLT_EPSILON)
			{
				dirX = floatRand() - 0.5;
				dirY = floatRand() - 0.5;
			}
			else
			{
				dirX = offsetX / dist;
				dirY = offsetY / dist;
			}

			float slope = calcSmoothingKernelDerivative(dist, smoothingRadius);

			float otherIndex_density = density[otherIndex];
			float sharedPressure = calcSharedPressure(otherIndex_density, density[particleIndex]);

			float a = sharedPressure * slope * mass[otherIndex] / (otherIndex_density + delta);

			pressureForce[0] += dirX * a;
			pressureForce[1] += dirY * a;
		}
	}

	float calcSmoothingKernel(float dist, float radius)
	{
		if (dist >= radius)
		{
			return 0;
		}

		float volume = (float)(M_PI * radius * radius * radius * radius) / 6;
		float influence = (radius - dist) * (radius - dist) / volume;
		return influence;
	}

	float calcSmoothingKernelDerivative(float dist, float radius)
	{
		if (dist >= radius)
		{
			return 0;
		}

		float scale = 12 / (float)(M_PI * radius * radius * radius * radius);
		float slope = (dist - radius) * scale;
		return slope;
	}

	float calcSharedPressure(float densityLeft, float densityRight) const
	{
		float pressureLeft = convertDensityToPressure(densityLeft);
		float pressureRight = convertDensityToPressure(densityRight);
		return (pressureLeft + pressureRight) * 0.5;
	}

        float convertDensityToPressure(float density) const
        {
                float densityError = density - targetDensity;
                float pressure = densityError * pressureMultiplier;
                return pressure;
        }

	void calcInteractionForce(float outInteractionForce[], int particleIndex) const
	{
		//戻り値を初期化
		outInteractionForce[0] = 0;
		outInteractionForce[1] = 0;

		ForcePoint p = this->forcePoint;

		float offsetX = p.pos[0] - this->pos[particleIndex][0];
		float offsetY = p.pos[1] - this->pos[particleIndex][1];

		float sqrDst = (offsetX * offsetX + offsetY * offsetY);
		if (!(sqrDst < p.radius * p.radius))
		{
			//影響範囲外
			//戻り値そのまま
			return;
		}

		float dirToForcePosX = 0;
		float dirToForcePosY = 0;

		float dist = std::sqrt(sqrDst);
		if (FLT_EPSILON < dist)
		{
			dirToForcePosX = offsetX / dist;
			dirToForcePosY = offsetY / dist;
		}

		float centreT = 1 - dist / p.radius;

		float velX = this->vel[particleIndex][0];
		float velY = this->vel[particleIndex][1];

		outInteractionForce[0] = (dirToForcePosX * p.strength - velX) * centreT;
		outInteractionForce[1] = (dirToForcePosY * p.strength - velY) * centreT;
	}

};



// グローバル変数:
HINSTANCE hInst;                                // 現在のインターフェイス
WCHAR szTitle[MAX_LOADSTRING];                  // タイトル バーのテキスト
WCHAR szWindowClass[MAX_LOADSTRING];            // メイン ウィンドウ クラス名

World world = World();
int canvasHeight = 900;

// このコード モジュールに含まれる関数の宣言を転送します:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

        // グローバル文字列を初期化する
	LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_WINDOWSPROJECTOPTIMIZATIONSPH, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// アプリケーション初期化の実行:
	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WINDOWSPROJECTOPTIMIZATIONSPH));

	MSG msg;


	// メイン メッセージ ループ:
	while (GetMessage(&msg, nullptr, 0, 0))
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return (int)msg.wParam;
}



//
//  関数: MyRegisterClass()
//
//  目的: ウィンドウ クラスを登録します。
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WINDOWSPROJECTOPTIMIZATIONSPH));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_WINDOWSPROJECTOPTIMIZATIONSPH);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassExW(&wcex);
}

//
//   関数: InitInstance(HINSTANCE, int)
//
//   目的: インスタンス ハンドルを保存して、メイン ウィンドウを作成します
//
//   コメント:
//
//        この関数で、グローバル変数でインスタンス ハンドルを保存し、
//        メイン プログラム ウィンドウを作成および表示します。
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // グローバル変数にインスタンス ハンドルを格納する

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

//
//  関数: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  目的: メイン ウィンドウのメッセージを処理します。
//
//  WM_COMMAND  - アプリケーション メニューの処理
//  WM_PAINT    - メイン ウィンドウを描画する
//  WM_DESTROY  - 中止メッセージを表示して戻る
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// 選択されたメニューの解析:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;

	case WM_LBUTTONUP:
	case WM_RBUTTONUP:
	{
		world.deleteInteractionForce();
	}
	break;

	case WM_LBUTTONDOWN:
	case WM_RBUTTONDOWN:
	case WM_MOUSEMOVE:
	{
		//ボタンを押しているか確認
		bool isPressedLeft = (GetKeyState(VK_LBUTTON) & 0x80);
		bool isPressedRight = (GetKeyState(VK_RBUTTON) & 0x80);

		if (!isPressedLeft && !isPressedRight)
		{
			break;
		}

		if (isPressedLeft && isPressedRight)
		{
			world.deleteInteractionForce();
			break;
		}

		float strength = 50 * (isPressedRight ? -1 : 1);

		int mouseX = LOWORD(lParam);
		int mouseY = HIWORD(lParam);

		float canvasWidth = (world.getWorldWidth() / world.getWorldHeight()) * canvasHeight;
		float perMouseX = mouseX / (float)canvasWidth;
		float perMouseY = mouseY / (float)canvasHeight;

		float worldMouseX = world.getWorldWidth() * perMouseX;
		float worldMouseY = world.getWorldHeight() * perMouseY;

		world.setInteractionForce(worldMouseX, worldMouseY, 2, strength);
	}
	break;

	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);

		// オフスクリーンバッファの作成
		HDC hdcOffscreen;
		HBITMAP hbm;
		BITMAPINFO bi;

		ZeroMemory(&bi, sizeof(bi));
		bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
		bi.bmiHeader.biWidth = ps.rcPaint.right - ps.rcPaint.left;
		bi.bmiHeader.biHeight = ps.rcPaint.bottom - ps.rcPaint.top;
		bi.bmiHeader.biPlanes = 1;
		bi.bmiHeader.biBitCount = 32;
		bi.bmiHeader.biCompression = BI_RGB;

		hbm = CreateCompatibleBitmap(hdc, bi.bmiHeader.biWidth, bi.bmiHeader.biHeight);

		// オフスクリーンバッファへの描画用のコンテキストを取得
		hdcOffscreen = CreateCompatibleDC(hdc);
		SelectObject(hdcOffscreen, hbm);

		// 背景の初期化
		HBRUSH hbr = CreateSolidBrush(RGB(255, 255, 255));
		FillRect(hdcOffscreen, &ps.rcPaint, hbr);

		// 図形の描画
		world.update(0.005);
		world.paint(hdcOffscreen, canvasHeight);

		// オフスクリーンバッファを画面に転送
		BitBlt(hdc, ps.rcPaint.left, ps.rcPaint.top, bi.bmiHeader.biWidth, bi.bmiHeader.biHeight, hdcOffscreen, 0, 0, SRCCOPY);
		DeleteObject(hbm);
		DeleteDC(hdcOffscreen);

		EndPaint(hWnd, &ps);

		InvalidateRect(hWnd, NULL, FALSE); // 再描画
	}
	break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// バージョン情報ボックスのメッセージ ハンドラーです。
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}
