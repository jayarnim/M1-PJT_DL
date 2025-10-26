# CTR Prediction with Deep Learning: Is There Really a Trade-off between Performance and Visibility & Linearity?

- Period: 2023.11.

- Professor: `Ahn,S.`

- Team: [`Wang,J.`](https://github.com/jayarnim)

## idea

Traditionally, data mining, which extracts information from data, has been conducted through statistical models. Statistical models have strong explanatory power for data that have already occurred, but have limitations in predictive power for future events. This duality originates from the linearity and explicitness of statistical models. In other words, while linearity and explicitness enable interpretability, the various assumptions and constraints required to satisfy them limit the ability to fully capture complex patterns in data.

On the other hand, recently emerging artificial neural network algorithms compensate for these limitations in predictive power. The basic MLP (Multi-Layer Perceptron) architecture combines multiple linear regression models with nonlinear activation functions, enabling effective learning of nonlinear patterns in data. However, since the inference process is performed implicitly, it is often called a “black-box” model, which entails a loss of explanatory power.

This project aimed to empirically examine the trade-off between linearity, explicitness, and predictive power using models from the Factorization Machines (FM) family. FM is mainly used in Click-Through Rate (CTR) prediction and is similar to a linear regression model that includes interaction terms between variables. The difference is that, while linear regression defines unique interaction weights for each variable pair, FM assigns an embedding vector to each variable and computes the interaction weight as the dot product between the embedding vectors. Through this approach, FM alleviates the sparsity problem that commonly occurs in CTR datasets.

Subsequent studies have developed in the direction of implementing interaction terms using neural networks: Neural Factorization Machine (NFM), Deep Factorization Machine (DeepFM), and eXtreme Deep Factorization Machine (xDeepFM). In summary, FM is a linear and explicit model, NFM is a nonlinear and implicit model, DeepFM is a hybrid model that uses both linear and nonlinear components, and xDeepFM is a nonlinear model that combines the implicit structure of DNNs with the explicit structure of CIN.

## models

### factorization machine series

- Factorization Machine:

$$
\hat{y}=\underbrace{f(X)}_{\text{main effect}} + \underbrace{g_{\mathrm{FM}}(X)}_{\text{interaction effect}}
$$

- Neural Factorization Machine (aggregate function is changed from element-wise product to concatenation):

$$
\hat{y}=\underbrace{f(X)}_{\text{main effect}} + \underbrace{g_{\mathrm{DNN}}(X)}_{\text{interaction effect}}
$$

- Deep Factorization Machine:

$$
\hat{y}=\underbrace{f(X)}_{\text{main effect}} + \underbrace{g_{\mathrm{FM}}(X) + g_{\mathrm{DNN}}(X)}_{\text{interaction effect}}
$$

- eXtreme Deep Factorization Machine:

$$
\hat{y}=\underbrace{f(X)}_{\text{main effect}} + \underbrace{g_{\mathrm{CIN}}(X) + g_{\mathrm{DNN}}(X)}_{\text{interaction effect}}
$$

### components

- main effect:

$$
f(X) = \beta_{0} + \sum_{i}{\beta_{i}x_{i}}
$$

- interaction effect with fm function:

$$
g_{\mathrm{FM}}(X) = \sum_{(i,j)}{\langle V_{i},V_{j}\rangle x_{i}x_{j}}
$$

- interaction effect with dnn function:

$$
g_{\mathrm{DNN}}(X) = \mathrm{MLP}_{\mathrm{ReLU}}\left(\left[\cdots \oplus V_{i}x_{i} \oplus V_{j}x_{j} \oplus \cdots \right]\right)
$$

- interaction effect with cin function:

$$\begin{aligned}
g_{\mathrm{CIN}}(X)&=\mathrm{Linear}\left(\left[H^{(1)} \oplus \cdots \oplus H^{(N)}\right]\right)\\
H^{(k)}&=\mathrm{SumPooling}\left(X^{(k)}\right)\\
X^{(k)}&=\mathrm{Conv1D}_{(1,1)}\left(\left[X^{(0)} \otimes X^{(k-1)}\right]\right)\\
X^{(0)}&=\begin{pmatrix}\cdots & V_{i}x_{i} & V_{j}x_{j} &\cdots \end{pmatrix}^{T}
\end{aligned}$$

## experiment

To evaluate the performance of the proposed model, the following dataset was used and split into `trn`, `val`, and `tst` sets in a 6:2:2 ratio, stratified by `user ID`:

- movielens latest small (2018) [`link`](https://grouplens.org/datasets/movielens/latest/)

This dataset contains various fields, such as `user ID`, `movie ID`, `timestamp` (when the rating was given), `movie title`, `movie genre`, and `explicit ratings` for user–movie pairs. I extracted year, month, day, and weekday from the timestamp, and year of release from the title. Finally, I used `user ID`, `movie ID`, the parsed timestamp components (`year`, `month`, `day`, `weekday`), `year of release`, and `movie genre` as explanatory variables, and the `explicit rating` as the response variable.

- factorization machine [`notebook`](/_notebooks/1.FM.ipynb)

- neural factorization machine [`notebook`](/_notebooks/2.NFM.ipynb)

- deep factorization machine [`notebook`](/_notebooks/3.DeepFM.ipynb)

- extreme deep factorization machine [`notebook`](/_notebooks/4.xDeepFM.ipynb)

Experimental results showed that DeepFM achieved the highest predictive performance. However, while FM maintained stable generalization even after many epochs without overfitting, the neural network–based models (NFM, DeepFM, and xDeepFM) experienced overfitting. To alleviate this, L2 regularization (Weight Decay), learning rate adjustments, and dropout were applied. As the weight decay increased, underfitting occurred; as the learning rate decreased, training stagnation was observed. Setting dropout to 0.5 showed some mitigation effect, but it did not completely resolve the overfitting problem.

This study empirically confirmed the trade-off between predictive power and linearity using the FM family models. FM, with its simple structure, was resistant to overfitting and provided clear interpretability, but its predictive power was limited. In contrast, NFM, DeepFM, and xDeepFM achieved higher predictive performance through nonlinearity, but overfitting due to model complexity emerged as a major limitation. Therefore, it can be concluded that improving predictive power involves increasing nonlinearity and model complexity, while sacrificing explicitness and generalizability.