<?php

declare(strict_types=1);

require_once __DIR__ . '/../bootstrap.php';

class PredictionService
{
    public static function isApiHealthy(): bool
    {
        $url = rtrim((string) config('python_api.base_url'), '/') . '/health';
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 5,
        ]);

        $response = curl_exec($ch);
        $error = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false || $error || $statusCode >= 400) {
            return false;
        }

        $decoded = json_decode($response, true);
        return is_array($decoded) && ($decoded['status'] ?? null) === 'ok';
    }

    public static function callPythonApi(string $queryType, string $inputText, int $topK = 10, string $dataset = 'C-dataset'): array
    {
        $url = rtrim((string) config('python_api.base_url'), '/') . '/predict';
        $payload = json_encode([
            'query_type' => $queryType,
            'input_text' => $inputText,
            'top_k' => $topK,
            'dataset' => $dataset,
        ], JSON_UNESCAPED_UNICODE);

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => $payload,
            CURLOPT_TIMEOUT => (int) config('python_api.timeout', 20),
        ]);

        $response = curl_exec($ch);
        $error = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if ($response === false || $error) {
            throw new RuntimeException('Không thể kết nối Python API: ' . $error);
        }

        if ($statusCode >= 400) {
            throw new RuntimeException('Python API trả lỗi HTTP ' . $statusCode);
        }

        $decoded = json_decode($response, true);
        if (!is_array($decoded)) {
            throw new RuntimeException('Dữ liệu trả về từ Python API không hợp lệ.');
        }

        return $decoded;
    }

    public static function saveHistory(int $userId, string $queryType, string $inputText, int $topK, array $apiResult): int
    {
        $matched = $apiResult['matched_input'] ?? [];
        $matchedSourceCode = $matched['id'] ?? null;
        $matchedType = $matched['type'] ?? null;
        $matchedInternalId = null;

        if ($matchedSourceCode && $matchedType) {
            $table = ($matchedType === 'drug') ? 'drugs' : 'diseases';
            $lookupStmt = db()->prepare("SELECT id FROM $table WHERE source_code = :code LIMIT 1");
            $lookupStmt->execute(['code' => $matchedSourceCode]);
            $row = $lookupStmt->fetch();
            if ($row) {
                $matchedInternalId = (int) $row['id'];
            }
        }

        $stmt = db()->prepare(
            'INSERT INTO prediction_requests (user_id, query_type, input_text, matched_entity_id, matched_entity_type, top_k, status, request_ip)
             VALUES (:user_id, :query_type, :input_text, :matched_entity_id, :matched_entity_type, :top_k, :status, :request_ip)'
        );

        $stmt->execute([
            'user_id' => $userId,
            'query_type' => $queryType,
            'input_text' => $inputText,
            'matched_entity_id' => $matchedInternalId,
            'matched_entity_type' => $matchedType,
            'top_k' => $topK,
            'status' => 'success',
            'request_ip' => $_SERVER['REMOTE_ADDR'] ?? '127.0.0.1',
        ]);

        $requestId = (int) db()->lastInsertId();

        $resultStmt = db()->prepare(
            'INSERT INTO prediction_results (request_id, rank_no, result_entity_type, result_entity_id, result_name, score, metadata_json)
             VALUES (:request_id, :rank_no, :result_entity_type, :result_entity_id, :result_name, :score, :metadata_json)'
        );

        foreach (($apiResult['results'] ?? []) as $index => $item) {
            $itemType = $item['type'] ?? 'disease';
            $itemSourceCode = $item['id'] ?? null;
            $itemInternalId = null;

            if ($itemSourceCode) {
                $itemTable = ($itemType === 'drug') ? 'drugs' : 'diseases';
                $itemLookup = db()->prepare("SELECT id FROM $itemTable WHERE source_code = :code LIMIT 1");
                $itemLookup->execute(['code' => $itemSourceCode]);
                $itemRow = $itemLookup->fetch();
                if ($itemRow) {
                    $itemInternalId = (int) $itemRow['id'];
                }
            }

            $resultStmt->execute([
                'request_id' => $requestId,
                'rank_no' => $index + 1,
                'result_entity_type' => $itemType,
                'result_entity_id' => $itemInternalId,
                'result_name' => $item['name'] ?? 'Unknown',
                'score' => $item['score'] ?? 0,
                'metadata_json' => json_encode($item, JSON_UNESCAPED_UNICODE),
            ]);
        }

        return $requestId;
    }
}
