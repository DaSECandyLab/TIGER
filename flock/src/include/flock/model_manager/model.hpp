#pragma once

#include "fmt/format.h"
#include <functional>
#include <utility>

#include "duckdb/main/connection.hpp"
#include "flock/core/common.hpp"
#include "flock/core/config.hpp"
#include "flock/model_manager/providers/adapters/anthropic.hpp"
#include "flock/model_manager/providers/adapters/azure.hpp"
#include "flock/model_manager/providers/adapters/ollama.hpp"
#include "flock/model_manager/providers/adapters/openai.hpp"
#include "flock/model_manager/providers/handlers/ollama.hpp"
#include "flock/model_manager/repository.hpp"
#include <nlohmann/json.hpp>

namespace flock {

enum class ExecutionMode {
    SYNC,
    ASYNC
};

class Model {
public:
    explicit Model(const nlohmann::json& model_json);
    explicit Model() = default;
    void AddCompletionRequest(const std::string& prompt, const int num_output_tuples, OutputType output_type = OutputType::STRING, const nlohmann::json& media_data = nlohmann::json::object());
    // 将 cacheblend 构造好的 token id prompt 继续透传到 provider。
    // 不启用 cacheblend 的查询仍然走 AddCompletionRequest 的原始文本路径。
    // ---add---
    void AddCompletionRequestTokenIds(const std::vector<int>& prompt_token_ids, const int num_output_tuples, OutputType output_type = OutputType::STRING, bool cacheblend = false, const std::string& blend_special_str = "");
    // ---add---
    void AddEmbeddingRequest(const std::vector<std::string>& inputs);
    void AddTranscriptionRequest(const nlohmann::json& audio_files);
    // 对外暴露 tokenizer，供 cacheblend 在执行阶段把每个 prompt 分段单独编码。
    std::vector<int> TokenizePrompt(const std::string& prompt, bool add_special_tokens);
    std::vector<nlohmann::json> CollectCompletions(const std::string& contentType = "application/json");
    std::vector<nlohmann::json> CollectEmbeddings(const std::string& contentType = "application/json");
    std::vector<nlohmann::json> CollectTranscriptions(const std::string& contentType = "multipart/form-data");
    ModelDetails GetModelDetails();
    nlohmann::json GetModelDetailsAsJson() const;

    // 供 binder 把模型详情解析为 JSON 的静态辅助方法。
    static nlohmann::json ResolveModelDetailsToJson(const nlohmann::json& user_model_json);

    // 创建 mock provider 的工厂函数类型。
    using MockProviderFactory = std::function<std::shared_ptr<IProvider>()>;

    // 设置工厂函数，为每个 Model 创建独立的 mock provider。
    static void SetMockProviderFactory(MockProviderFactory factory) {
        mock_provider_factory_ = std::move(factory);
    }

    // 旧接口：设置共享 mock provider，用于兼容旧测试；并行测试下安全性较弱。
    static void SetMockProvider(const std::shared_ptr<IProvider>& mock_provider) {
        mock_provider_ = mock_provider;
    }

    static void ResetMockProvider() {
        mock_provider_ = nullptr;
        mock_provider_factory_ = nullptr;
    }

    std::shared_ptr<IProvider>
            provider_;

private:
    ModelDetails model_details_;
    inline static std::shared_ptr<IProvider> mock_provider_ = nullptr;
    inline static MockProviderFactory mock_provider_factory_ = nullptr;
    void ConstructProvider();
    void LoadModelDetails(const nlohmann::json& model_json);
    static std::tuple<std::string, std::string, nlohmann::basic_json<>> GetQueriedModel(const std::string& model_name);
    std::string GetSecret(const std::string& secret_name);
};

}// namespace flock
