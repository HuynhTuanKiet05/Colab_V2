<?php
require_once __DIR__ . '/../app/services/PredictionService.php';
require_login();

$user = current_user();
$resultData = null;
$error = null;
$queryType = $_POST['query_type'] ?? 'drug_to_disease';
$inputText = trim($_POST['input_text'] ?? '');
$dataset = $_POST['dataset'] ?? 'C-dataset';
$topK = max(1, min(20, (int) ($_POST['top_k'] ?? 10)));
$apiHealthy = PredictionService::isApiHealthy();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if ($inputText === '') {
        $error = 'Vui lòng nhập tên thuốc hoặc bệnh.';
    } elseif (!$apiHealthy) {
        $error = 'Hệ thống AI hiện đang ngoại tuyến. Vui lòng kiểm tra lại server.';
    } else {
        try {
            $resultData = PredictionService::callPythonApi($queryType, $inputText, $topK, $dataset);
            
            // Enrich results with names from local database if missing/ID-only
            if (!empty($resultData['results'])) {
                foreach ($resultData['results'] as &$item) {
                    if (($item['type'] ?? '') === 'disease') {
                        $stmt = db()->prepare('SELECT name FROM diseases WHERE source_code = :code LIMIT 1');
                        $stmt->execute(['code' => $item['id']]);
                        $dbDisease = $stmt->fetch();
                        if ($dbDisease) {
                            $item['name'] = $dbDisease['name'];
                        }
                    }
                }
            }
            unset($item);

            // Enrich graph node labels as well
            if (!empty($resultData['graph']['nodes'])) {
                foreach ($resultData['graph']['nodes'] as &$node) {
                    if (($node['type'] ?? '') === 'disease') {
                        $stmt = db()->prepare('SELECT name FROM diseases WHERE source_code = :code LIMIT 1');
                        $stmt->execute(['code' => $node['actual_id'] ?? $node['id']]);
                        $dbDisease = $stmt->fetch();
                        if ($dbDisease) {
                            $node['label'] = $dbDisease['name'];
                        }
                    }
                }
            }
            unset($node);

            PredictionService::saveHistory((int) $user['id'], $queryType, $inputText, $topK, $resultData);
            flash('success', 'Thành công');
            $_SESSION['latest_graph'] = $resultData['graph'] ?? ['nodes' => [], 'links' => []];
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$graph = $resultData['graph'] ?? ($_SESSION['latest_graph'] ?? ['nodes' => [], 'links' => []]);

$historyStmt = db()->prepare('SELECT * FROM prediction_requests WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 8');
$historyStmt->execute(['user_id' => $user['id']]);
$histories = $historyStmt->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HGT AI Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <script src="https://unpkg.com/3d-force-graph"></script>
    <style>
        .loading-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(8px);
            z-index: 9999;
            place-items: center;
            color: #fff;
            font-weight: 600;
        }
    </style>
</head>
<body>
<div id="loader" class="loading-overlay">Đang phân tích đồ thị...</div>

<div id="toast-container" class="toast-container"></div>

<div class="container">
    <div class="navbar">
        <div class="brand">AMNTDDA <span style="font-weight: 300; font-size: 0.7em; opacity: 0.6;">HGT Link Predictor</span></div>
        <div class="nav-links">
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="history.php">Lịch sử</a>
            <?php if (($user['role'] ?? '') === 'admin'): ?>
                <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="admin.php">Quản trị</a>
            <?php endif; ?>
            <a class="btn btn-danger" style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171;" href="logout.php">Đăng xuất</a>
        </div>
    </div>


    <?php if (!$apiHealthy): ?><div class="alert alert-error">Cảnh báo: Python HGT-API (Port 8000) đang ngắt kết nối.</div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>

    <div class="glass-card search-container" style="margin-bottom: 24px;">
        <h2 style="margin-bottom: 20px; font-size: 1.25rem;">Hệ thống dự đoán liên kết Thuốc - Bệnh</h2>
        <form method="post" onsubmit="document.getElementById('loader').style.display='grid'">
            <div class="search-grid">
                <div class="form-group">
                    <label class="label">Bộ dữ liệu</label>
                    <select class="select" name="dataset">
                        <option value="C-dataset" <?= $dataset === 'C-dataset' ? 'selected' : '' ?>>Dataset C (Chuẩn)</option>
                        <option value="B-dataset" <?= $dataset === 'B-dataset' ? 'selected' : '' ?>>Dataset B</option>
                        <option value="F-dataset" <?= $dataset === 'F-dataset' ? 'selected' : '' ?>>Dataset F</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label">Từ khóa (Tên hoặc ID)</label>
                    <input class="input" type="text" name="input_text" value="<?= e($inputText) ?>" placeholder="Goserelin, DB00014, D102100..." required>
                </div>
                <div class="form-group">
                     <label class="label" style="text-align: center;">Thực thể tìm</label>
                     <div class="query-toggle-container">
                        <input type="checkbox" id="query_toggle" class="toggle-input" name="query_type_toggle" <?= $queryType === 'disease_to_drug' ? 'checked' : '' ?> onchange="this.nextElementSibling.nextElementSibling.value = this.checked ? 'disease_to_drug' : 'drug_to_disease'">
                        <label for="query_toggle" class="toggle-label" title="Trượt lên: Thuốc, Trượt xuống: Bệnh">
                            <div class="toggle-slider"></div>
                            <div class="toggle-text">
                                <span class="text-top">Thuốc</span>
                                <span class="text-bottom">Bệnh</span>
                            </div>
                        </label>
                        <input type="hidden" name="query_type" value="<?= e($queryType) ?>">
                     </div>
                </div>
                <div class="form-group">
                    <label class="label">Lấy Top-K</label>
                    <div style="display: flex; gap: 8px;">
                        <input class="input input-top-k" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                        <button class="btn" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Dự đoán</button>
                    </div>
                </div>
            </div>
        </form>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <h3>Kết quả phân tích HGT</h3>
            <?php if ($resultData): ?>
                <div class="table-container">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Hạng</th>
                                <th>Thực thể tiềm năng</th>
                                <th>Loại</th>
                                <th>Xác suất (Score)</th>
                            </tr>
                        </thead>
                        <tbody>
                        <?php foreach (($resultData['results'] ?? []) as $index => $item): ?>
                            <tr>
                                <td style="font-weight: 600; opacity: 0.6;">#<?= $index + 1 ?></td>
                                <td>
                                    <div style="font-weight: 700; color: #fff;"><?= e((string) $item['name']) ?></div>
                                    <div style="font-size: 0.8em; opacity: 0.5; font-family: 'JetBrains Mono', monospace;">ID: <?= e((string) $item['id']) ?></div>
                                </td>
                                <td><span class="badge <?= ($item['type'] ?? '') === 'drug' ? 'badge-drug' : 'badge-disease' ?>"><?= e((string) $item['type']) ?></span></td>
                                <td class="score-text"><?= e(number_format((float) $item['score'], 4)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            <?php else: ?>
                <div style="padding: 60px 0; text-align: center; color: var(--text-muted);">
                    <p>Nhập từ khóa và nhấn dự đoán để bắt đầu phân tích.</p>
                </div>
            <?php endif; ?>
        </div>

        <div class="glass-card">
            <h3>Lịch sử gần đây</h3>
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Input</th>
                            <th>Kết quả</th>
                        </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($histories)): ?>
                        <tr><td colspan="2" style="text-align: center;" class="muted">Chưa có dữ liệu</td></tr>
                    <?php endif; ?>
                    <?php foreach ($histories as $item): ?>
                        <tr>
                            <td><span class="muted" style="font-size: 11px;"><?= e((string) $item['query_type']) ?></span><br><strong><?= e((string) $item['input_text']) ?></strong></td>
                            <td style="text-align: right;"><span class="badge" style="background: rgba(255,255,255,0.05);">Top-<?= e((string) $item['top_k']) ?></span></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="glass-card" style="margin-top:24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3>Đồ thị tương tác 3D</h3>
            <div class="muted">
                <span style="display:inline-block; width:10px; height:10px; background:#2563eb; border-radius:50%; margin-right:4px;"></span> Thuốc
                <span style="display:inline-block; width:10px; height:10px; background:#dc2626; border-radius:50%; margin-left:12px; margin-right:4px;"></span> Bệnh
                <span style="display:inline-block; width:10px; height:10px; background:#f59e0b; border-radius:50%; margin-left:12px; margin-right:4px;"></span> Protein
            </div>
        </div>
        
        <?php if (empty($graph['nodes'])): ?>
            <div style="height: 400px; display: grid; place-items: center; border: 2px dashed rgba(255,255,255,0.05); border-radius: 16px; color: var(--text-muted);">
                Chưa có dữ liệu đồ thị. Vui lòng thực hiện dự đoán.
            </div>
        <?php else: ?>
            <div id="graph3d"></div>
        <?php endif; ?>
    </div>
</div>

<script>
const graphData = <?= json_encode($graph, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const apiNote = <?= json_encode($resultData['note'] ?? '', JSON_UNESCAPED_UNICODE) ?>;
const phpSuccess = <?= json_encode($success ?? '', JSON_UNESCAPED_UNICODE) ?>;

// Toast logic
function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const isWarn = message.includes('⚠️');
    const isSuccess = message === 'Thành công';
    
    toast.className = `toast ${isWarn ? 'toast-warn' : ''} ${isSuccess ? 'toast-success' : ''}`;
    toast.innerHTML = `
        <div class="toast-icon">${isWarn ? '!' : (isSuccess ? '✓' : 'i')}</div>
        <div class="toast-message">${message}</div>
    `;
    
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 100);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

// Trigger toast if there is a warning or specific note
if (apiNote && (apiNote.includes('⚠️') || apiNote.includes('ℹ️'))) {
    showToast(apiNote);
} else if (phpSuccess) {
    showToast(phpSuccess);
}

const container = document.getElementById('graph3d');
if (container && graphData && Array.isArray(graphData.nodes) && graphData.nodes.length > 0) {
    const Graph = ForceGraph3D()(container)
        .backgroundColor('#00000000')
        .width(container.offsetWidth)
        .height(container.offsetHeight)
        .graphData(graphData)
        .nodeLabel(node => `<div style="background: rgba(0,0,0,0.85); padding: 12px; border-radius: 12px; border: 1px solid ${node.color}; backdrop-filter: blur(5px); box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="color: ${node.color}; font-weight: 800; font-size: 1.1em; margin-bottom: 4px;">${node.label}</div>
            <div style="font-size: 0.85em; color: #fff; opacity: 0.8; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;">ID: ${node.actual_id || node.id}</div>
            <div style="font-size: 0.8em; color: #ccc; text-transform: uppercase; letter-spacing: 0.05em;">Loại: ${node.type}</div>
        </div>`)
        .nodeColor(node => node.color || '#ffffff')
        .nodeOpacity(1)
        .nodeRelSize(7)
        .linkWidth(link => Math.max(1.5, (link.score || 0.5) * 6))
        .linkDirectionalParticles(3)
        .linkDirectionalParticleSpeed(0.006)
        .linkColor(() => 'rgba(255,255,255,0.4)')
        .linkOpacity(0.7);

    window.addEventListener('resize', () => {
        Graph.width(container.offsetWidth);
        Graph.height(container.offsetHeight);
    });
}
</script>
</body>
</html>
