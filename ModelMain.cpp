#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Windows.h>
#include <cmath>
#include <random>

#define ALL 1000001      // �ʻ���С
#define MAXN 10          // ����ά��
#define MAX_INPUT 1024   // ������󳤶�
#define CONTEXT_LENGTH 1024 // �����ĳ���
#define COMMA_MARKER "<CMA>" // �����������

using namespace std;

float WordVector[ALL][MAXN]; // ������
float Weights[ALL][MAXN];    // Ȩ��
string TokenList[ALL];        // �ʻ��
int alltoken = -1;            // ��ǰ�ʻ���С

// ��������
void LoadModel(const string& wvpath, const string& wghspath);
void Multi_Point_Attention();
void Softmax();
void AddNewToken(const string& token);
float stringToFloat(const string& str);
double probability[ALL];
int underprocess[CONTEXT_LENGTH]; // ������token����
int currentIndex = 0; // ��ǰ��������

// �����������
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.1, 0.1); // �����ʼ����Χ

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// �滻�ַ����еı�Ƿ���Ϊ����
string restoreComma(const string& str) {
    string result = str;
    size_t pos = 0;
    while ((pos = result.find(COMMA_MARKER, pos)) != string::npos) {
        result.replace(pos, strlen(COMMA_MARKER), ",");
        pos++; // �������滻�Ķ���
    }
    return result;
}

// �ַ���ת������
float stringToFloat(const string& str) {
    return stof(str);
}

void LoadModel(const string& wvpath, const string& wghspath) {
    ifstream wvfile(wvpath);
    if (!wvfile.is_open()) {
        cout << "�޷��򿪴������ļ���" << endl;
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
    cout << wvpath << " ��ȡ�ɹ���" << endl;

    ifstream wghsfile(wghspath);
    if (!wghsfile.is_open()) {
        cout << "�޷���Ȩ���ļ���" << endl;
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
    cout << wghspath << " ��ȡ�ɹ���" << endl;
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

    // Softmax���ʼ���
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

    cout << "���������ɵ��ı������ģ��Կո�ָ��� tokens�����س���������\n";
    string inputLine;
    getline(cin, inputLine);
    istringstream ss(inputLine);
    string token;

    // �����û�����
    while (ss >> token) {
        // ��� token �Ƿ����� TokenList ��
        bool tokenExists = false;
        for (int j = 0; j <= alltoken; j++) {
            if (token == TokenList[j]) {
                underprocess[currentIndex] = j; // ��������������
                tokenExists = true;
                break;
            }
        }

        // ��� token �����ڣ�������� token
        if (!tokenExists) {
            AddNewToken(token);
            underprocess[currentIndex] = alltoken; // ����ӵ� token ����
        }

        currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // ѭ����������
    }

    // �����ı�
    cout << "���ɵ��ı���";
    for (int i = 0; i < 10; i++) { // ����10�� token
        Multi_Point_Attention();
        
        // �ҵ��������� token
        double max_prob = -1;
        int pos = -1;
        for (int i = 0; i <= alltoken; i++) {
            if (probability[i] > max_prob) {
                max_prob = probability[i];
                pos = i;
            }
        }

        // ������ɵ� token
        cout << TokenList[pos] << " ";
        
        // ��������������
        underprocess[currentIndex] = pos; // ���µ�ǰ�����ĵ����һ�� token
        currentIndex = (currentIndex + 1) % CONTEXT_LENGTH; // ѭ����������
    }
    cout << endl;

    return 0;
}

