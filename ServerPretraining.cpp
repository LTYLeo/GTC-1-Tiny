#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Windows.h>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>

#define ALL 1000001 // 词汇表大小
#define MAXN 10     // 向量维度
#define MAX_TEXT 100 // 生成文本最大长度
#define MAX_INPUT 1024 // 输入最大长度
#define LEARNING_RATE 0.001 // 学习率
#define LAMBDA 0.001 // 正则化强度
#define MAX_GRADIENT 5.0 // 梯度裁剪阈值
#define CONTEXT_LENGTH 1024 // 上下文长度修改为1024
#define COMMA_MARKER "<CMA>" // 逗号替代符号

using namespace std;

// 用 std::vector 替代静态数组
vector<vector<float>> WordVector(ALL, vector<float>(MAXN, 0.0f));
vector<vector<float>> Weights(ALL, vector<float>(MAXN, 0.0f));
vector<string> TokenList(ALL);
int alltoken = -1;

// 函数声明
void Dispose_wordvector(vector<string> line_data);
void Dispose_weights(vector<string> line_data);
void AddNewToken(const string& token);
float stringToFloat(const string& str);
float f1[MAXN], f2[MAXN];
float h1[MAXN], h2[MAXN];
double probability[ALL];
double pro1[ALL];
double pro2[ALL];
int cnt = -1;
int underprocess[CONTEXT_LENGTH]; // 上下文token数组
int pos;
int currentIndex = 0; // 当前填充的索引

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.1, 0.1); // 随机初始化范围

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// 替换字符串中的逗号为标记符号
string replaceComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(",", pos)) != string::npos) {
        result.replace(pos, 1, COMMA_MARKER);
        pos += strlen(COMMA_MARKER);
    }
    return result;
}

// 替换字符串中的标记符号为逗号
string restoreComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(COMMA_MARKER, pos)) != string::npos) {
        result.replace(pos, strlen(COMMA_MARKER), ",");
        pos++;
    }
    return result;
}

// 字符串转浮点数
float stringToFloat(const string& str) {
    return stof(str);
}

// 日志记录函数
void LogProgress(const string& logPath, const string& message) {
    ofstream logFile(logPath, ios_base::app);
    if (!logFile.is_open()) {
        cerr << "无法写入日志文件！" << endl;
        return;
    }

    logFile << message << endl;
    logFile.close();
}

// 保存检查点
void SaveCheckpoint(const string& checkpointPath, const string& wvpath, const string& wghspath) {

    ofstream checkpointFile(checkpointPath);
    if (!checkpointFile.is_open()) {
        cerr << "无法保存检查点文件！" << endl;
        return;
    }

    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    checkpointFile << "Checkpoint at " << ctime(&now_time);

    checkpointFile.close();
    cout << "检查点保存成功。" << endl;
}

void Multi_Point_Attention() {
    // Feature 1
    memset(f1, 0, sizeof(f1));
    for (int i = 0; i < MAXN; i++) {
        for (int j = 0; j <= cnt; j++) {
            f1[i] += Weights[underprocess[j]][i] * WordVector[underprocess[j]][i];
        }
    }
    for (int i = 0; i < MAXN; i++) {
        f1[i] /= 100;
    }
    for (int i = 0; i < MAXN; i++) {
        h1[i] = sigmoid(f1[i]);
    }

    // Feature 2
    memset(f2, 0, sizeof(f2));
    for (int i = 0; i < MAXN; i++) {
        for (int j = 0; j <= cnt; j++) {
            for (int k = 0; k <= cnt; k++) {
                f2[i] += Weights[underprocess[j]][i] * WordVector[underprocess[k]][i];
            }
        }
    }
    for (int i = 0; i < MAXN; i++) {
        f2[i] /= (cnt + 1);
    }
    for (int i = 0; i < MAXN; i++) {
        f2[i] /= 100;
    }
    for (int i = 0; i < MAXN; i++) {
        h2[i] = sigmoid(f2[i]);
    }
}

void Softmax() {
    // pro1
    double sum_exp = 0;
    for (int i = 0; i <= alltoken; i++) {
        double dot_product = 0;
        for (int j = 0; j < MAXN; j++) {
            dot_product += h1[j] * WordVector[i][j];
        }
        pro1[i] = exp(dot_product);
        sum_exp += pro1[i];
    }
    for (int i = 0; i <= alltoken; i++) {
        pro1[i] /= sum_exp;
    }

    // pro2
    sum_exp = 0;
    for (int i = 0; i <= alltoken; i++) {
        double dot_product = 0;
        for (int j = 0; j < MAXN; j++) {
            dot_product += h2[j] * WordVector[i][j];
        }
        pro2[i] = exp(dot_product);
        sum_exp += pro2[i];
    }
    for (int i = 0; i <= alltoken; i++) {
        pro2[i] /= sum_exp;
    }

    // probability
    for (int i = 0; i <= alltoken; i++) {
        probability[i] = (pro1[i] + pro2[i]) / 2;
    }
}

void Dispose_wordvector(vector<string> line_data) {
    int tot = 0;
    int position;
    for (string str : line_data) {
        tot++;
        if (tot == 1) {
            position = stoi(str);
            continue;
        } else if (tot == 2) {
            TokenList[position] = restoreComma(str);
            continue;
        } else {
            WordVector[position][tot - 3] = stringToFloat(str); // 使用新函数
            continue;
        }
    }
    alltoken++;
}

void Dispose_weights(vector<string> line_data) {
    int tot = 0;
    int position;
    for (string str : line_data) {
        tot++;
        if (tot == 1) {
            position = stoi(str);
            continue;
        } else {
            Weights[position][tot - 2] = stringToFloat(str); // 使用新函数
            continue;
        }
    }
}

void AddNewToken(const string& token) {
    alltoken++;
    TokenList[alltoken] = token;

    // 随机初始化 WordVector
    for (int j = 0; j < MAXN; j++) {
        WordVector[alltoken][j] = dis(gen);
    }

    // 随机初始化 Weights
    for (int j = 0; j < MAXN; j++) {
        Weights[alltoken][j] = dis(gen);
    }
}

void SaveModel(const string& wvpath, const string& wghspath) {
    ofstream wvfile(wvpath);
    if (!wvfile.is_open()) {
        cout << "无法保存词向量文件！" << endl;
        return;
    }

    // 保存 WordVector，保持格式一致
    for (int i = 0; i <= alltoken; i++) {
        wvfile << i << "," << replaceComma(TokenList[i]); // 写入index和token
        for (int j = 0; j < MAXN; j++) {
            wvfile << "," << WordVector[i][j];
        }
        wvfile << endl;
    }
    wvfile.close();
    cout << wvpath << " 保存成功。" << endl;

    ofstream wghsfile(wghspath);
    if (!wghsfile.is_open()) {
        cout << "无法保存权重文件！" << endl;
        return;
    }

    // 保存 Weights，保持格式一致
    for (int i = 0; i <= alltoken; i++) {
        wghsfile << i; // 写入index
        for (int j = 0; j < MAXN; j++) {
            wghsfile << "," << Weights[i][j];
        }
        wghsfile << endl;
    }
    wghsfile.close();
    cout << wghspath << " 保存成功。" << endl;
}

int main() {
    SetConsoleTitle("GTC-1 Tiny (gtc-1-tiny-build-005-CMAfixed)");
    cout << "Loading Data..." << endl;
    string wvpath = "WordVector_Tiny.csv";
    string wghspath = "Weights_Tiny.csv";
    string logPath = "training_log.txt";
    string checkpointPath = "checkpoint.txt";

    ifstream wvfile(wvpath, ios::in);
    ifstream wghsfile(wghspath, ios::in);

    // 读取词向量
    if (!wvfile.is_open()) {
        cout << "Data Crush!" << endl;
        exit(1);
    } else {
        string line;
        vector<string> words;
        string word;
        getline(wvfile, line);
        istringstream strin;
        while (getline(wvfile, line)) {
            words.clear();
            strin.clear();
            strin.str(line);
            while (getline(strin, word, ',')) {
                words.push_back(word);
            }
            Dispose_wordvector(words);
        }
        wvfile.close();
    }
    cout << wvpath << " Read OK." << endl;

    // 读取权重
    if (!wghsfile.is_open()) {
        cout << "Data Crush!" << endl;
        exit(1);
    } else {
        string line;
        vector<string> words;
        string word;
        getline(wghsfile, line);
        istringstream strin;
        while (getline(wghsfile, line)) {
            words.clear();
            strin.clear();
            strin.str(line);
            while (getline(strin, word, ',')) {
                words.push_back(word);
            }
            Dispose_weights(words);
        }
        wghsfile.close();
    }
    cout << wghspath << " Read OK." << endl;

    // 读取训练数据
    string trainFile = "TrainData.txt";
    ifstream trainData(trainFile);
    if (!trainData.is_open()) {
        cout << "无法打开训练文件！" << endl;
        exit(1);
    }

    string line;
    int tokenProcessedCount = 0; // 计数器，用于跟踪已处理的token数量
    while (getline(trainData, line)) {
        vector<string> tokens;
        istringstream ss(line);
        string token;
        
        // 使用空格分隔并提取非空 token
        while (ss >> token) {
            if (!token.empty()) {
                tokens.push_back(token); // 只添加非空 token
            }
        }

        cnt = tokens.size() - 1; // 记录输入 token 数量
        for (int i = 0; i < tokens.size(); i++) {
            string s = tokens[i];
            
            // 检查 token 是否已在 TokenList 中
            bool tokenExists = false;
            for (int j = 0; j <= alltoken; j++) {
                if (s == TokenList[j]) {
                    underprocess[currentIndex] = j; // 更新上下文数组
                    tokenExists = true;
                    break;
                }
            }
            
            // 如果 token 不存在，则添加新 token
            if (!tokenExists) {
                AddNewToken(s);
                underprocess[currentIndex] = alltoken; // 新添加的 token 索引
            }

            // 预测下一个 token
            if (i > 0) { // 进行多轮训练
                double loss = 1.0;
                for (int step = 0; step < 6000; step++) {
                    Multi_Point_Attention();
                    Softmax();

                    // 计算损失，例如通过交叉熵损失
                    loss = -log(probability[underprocess[currentIndex]]);
                    // 添加正则化损失
                    double regularization_loss = 0.0;
                    for (int j = 0; j < MAXN; j++) {
                        regularization_loss += (Weights[underprocess[currentIndex]][j] * Weights[underprocess[currentIndex]][j]);
                    }
                    loss += LAMBDA * regularization_loss;

                    cout << "Step: " << step + 1 << " Loss: " << loss << endl;

                    // 更新模型权重
                    for (int j = 0; j < MAXN; j++) {
                        float gradient = probability[underprocess[currentIndex]] - 1; // 假设这是计算出来的梯度
                        
                        // 梯度裁剪
                        if (gradient > MAX_GRADIENT) {
                            gradient = MAX_GRADIENT;
                        } else if (gradient < -MAX_GRADIENT) {
                            gradient = -MAX_GRADIENT;
                        }

                        Weights[underprocess[currentIndex]][j] -= LEARNING_RATE * (gradient + LAMBDA * Weights[underprocess[currentIndex]][j]);
                        WordVector[underprocess[currentIndex]][j] -= LEARNING_RATE * (gradient + LAMBDA * WordVector[underprocess[currentIndex]][j]);
                    }

                    // 如果损失小于0.1，退出循环
                    if (loss <= 0.1) {
                        break;
                    }
                }

                // 找到概率最大的 token
                double maxx = -1;
                for (int i = 0; i <= alltoken; i++) {
                    if (probability[i] > maxx) {
                        maxx = probability[i];
                        pos = i;
                    }
                }
                // 输出生成的 token
                cout << " " << TokenList[pos];
                if (TokenList[pos] == "<TGC>") break; // 结束条件
                
                // 更新上下文数组
                currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // 循环更新索引
                underprocess[currentIndex] = pos; // 更新当前上下文的最后一个token

                // 更新处理的 token 计数
                tokenProcessedCount++;

                // 每5个token保存一次参数和检查点
                if (tokenProcessedCount % 5 == 0) {
                	SaveModel(wvpath, wghspath);
                    SaveCheckpoint(checkpointPath, wvpath, wghspath);
                    string logMessage = "Processed " + to_string(tokenProcessedCount) + " tokens.";
                    LogProgress(logPath, logMessage);
                    tokenProcessedCount = 0;
                }
            }
        }
        cout << endl;
    }

    trainData.close();
    // 保存模型
    SaveModel(wvpath, wghspath); // 保存更新后的模型参数
    return 0;
}

