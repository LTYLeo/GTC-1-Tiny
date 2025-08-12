#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Windows.h>
#include <cmath>
#include <random>

#define ALL 1000001      // 词汇表大小
#define MAXN 10          // 向量维度
#define MAX_INPUT 1024   // 输入最大长度
#define CONTEXT_LENGTH 1024 // 上下文长度
#define COMMA_MARKER "<CMA>" // 逗号替代符号

using namespace std;

float WordVector[ALL][MAXN]; // 词向量
float Weights[ALL][MAXN];    // 权重
string TokenList[ALL];        // 词汇表
int alltoken = -1;            // 当前词汇表大小

// 函数声明
void LoadModel(const string& wvpath, const string& wghspath);
void Multi_Point_Attention();
void Softmax();
void AddNewToken(const string& token);
float stringToFloat(const string& str);
double probability[ALL];
int underprocess[CONTEXT_LENGTH]; // 上下文token数组
int currentIndex = 0; // 当前填充的索引

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.1, 0.1); // 随机初始化范围

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// 替换字符串中的标记符号为逗号
string restoreComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(COMMA_MARKER, pos)) != string::npos) {
        result.replace(pos, strlen(COMMA_MARKER), ",");
        pos++; // 跳过刚替换的逗号
    }
    return result;
}

// 字符串转浮点数
float stringToFloat(const string& str) {
    return stof(str);
}

void LoadModel(const string& wvpath, const string& wghspath) {
    ifstream wvfile(wvpath);
    if (!wvfile.is_open()) {
        cout << "无法打开词向量文件！" << endl;
        exit(1);
    }
    string line;
    vector<string> words;
    while (getline(wvfile, line)) {
        istringstream strin(line);
        words.clear();
        string word;
        while (getline(strin, word, ',')) {
            words.push_back(word);
        }
        int position = stoi(words[0]);
        TokenList[position] = restoreComma(words[1]);
        for (int i = 0; i < MAXN; i++) {
            WordVector[position][i] = stringToFloat(words[i + 2]);
        }
        alltoken++;
    }
    wvfile.close();
    cout << wvpath << " 读取成功。" << endl;

    ifstream wghsfile(wghspath);
    if (!wghsfile.is_open()) {
        cout << "无法打开权重文件！" << endl;
        exit(1);
    }
    while (getline(wghsfile, line)) {
        istringstream strin(line);
        words.clear();
        string word;
        while (getline(strin, word, ',')) {
            words.push_back(word);
        }
        int position = stoi(words[0]);
        for (int i = 0; i < MAXN; i++) {
            Weights[position][i] = stringToFloat(words[i + 1]);
        }
    }
    wghsfile.close();
    cout << wghspath << " 读取成功。" << endl;
}

void Multi_Point_Attention() {
    float f1[MAXN] = {0};
    float f2[MAXN] = {0};
    float h1[MAXN], h2[MAXN];

    // Feature 1
    for (int i = 0; i <= currentIndex; i++) {
        for (int j = 0; j < MAXN; j++) {
            f1[j] += Weights[underprocess[i]][j] * WordVector[underprocess[i]][j];
        }
    }
    for (int i = 0; i < MAXN; i++) {
        h1[i] = sigmoid(f1[i]);
    }

    // Feature 2
    for (int i = 0; i <= currentIndex; i++) {
        for (int j = 0; j <= currentIndex; j++) {
            for (int k = 0; k < MAXN; k++) {
                f2[k] += Weights[underprocess[i]][k] * WordVector[underprocess[j]][k];
            }
        }
    }
    for (int i = 0; i < MAXN; i++) {
        f2[i] /= (currentIndex + 1);
        h2[i] = sigmoid(f2[i]);
    }

    // Softmax概率计算
    double sum_exp = 0;
    for (int i = 0; i <= alltoken; i++) {
        double dot_product = 0;
        for (int j = 0; j < MAXN; j++) {
            dot_product += h1[j] * WordVector[i][j];
        }
        probability[i] = exp(dot_product);
        sum_exp += probability[i];
    }
    for (int i = 0; i <= alltoken; i++) {
        probability[i] /= sum_exp;
    }
}

int main() {
    SetConsoleTitle("GTC-1 Text Generation");
    
    string wvpath = "WordVector_Tiny.csv";
    string wghspath = "Weights_Tiny.csv";
    LoadModel(wvpath, wghspath);

    cout << "请输入生成的文本上下文（以空格分隔的 tokens，按回车结束）：\n";
    string inputLine;
    getline(cin, inputLine);
    istringstream ss(inputLine);
    string token;

    // 处理用户输入
    while (ss >> token) {
        // 检查 token 是否已在 TokenList 中
        bool tokenExists = false;
        for (int j = 0; j <= alltoken; j++) {
            if (token == TokenList[j]) {
                underprocess[currentIndex] = j; // 更新上下文数组
                tokenExists = true;
                break;
            }
        }

        // 如果 token 不存在，则添加新 token
        if (!tokenExists) {
            AddNewToken(token);
            underprocess[currentIndex] = alltoken; // 新添加的 token 索引
        }

        currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // 循环更新索引
    }

    // 生成文本
    cout << "生成的文本：";
    for (int i = 0; i < 10; i++) { // 生成10个 token
        Multi_Point_Attention();
        
        // 找到概率最大的 token
        double max_prob = -1;
        int pos = -1;
        for (int i = 0; i <= alltoken; i++) {
            if (probability[i] > max_prob) {
                max_prob = probability[i];
                pos = i;
            }
        }

        // 输出生成的 token
        cout << TokenList[pos] << " ";
        
        // 更新上下文数组
        underprocess[currentIndex] = pos; // 更新当前上下文的最后一个 token
        currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // 循环更新索引
    }
    cout << endl;

    return 0;
}

