<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

$stats = [
    'drugs' => (int) db()->query('SELECT COUNT(*) FROM drugs')->fetchColumn(),
    'diseases' => (int) db()->query('SELECT COUNT(*) FROM diseases')->fetchColumn(),
    'proteins' => (int) db()->query('SELECT COUNT(*) FROM proteins')->fetchColumn(),
    'predictions' => (int) db()->query('SELECT COUNT(*) FROM prediction_requests')->fetchColumn(),
    'links' => (int) db()->query('SELECT COUNT(*) FROM drug_disease_links')->fetchColumn(),
];

$recent = db()->query('SELECT * FROM prediction_requests ORDER BY created_at DESC LIMIT 8')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Hệ thống quản trị</title>
    <link rel="stylesheet" href="assets/style.css">
    <style>
        .admin-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card h2 {
            font-size: 2.5rem;
            margin-top: 10px;
            background: linear-gradient(to right, #fff, var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="brand">Admin <span style="font-weight: 300; opacity: 0.6;">Panel</span></div>
        <div class="nav-links">
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="index.php">Dashboard</a>
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="admin_drugs.php">Quản lý thuốc</a>
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="admin_diseases.php">Quản lý bệnh</a>
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="admin_links.php">Quản lý liên kết</a>
            <a class="btn btn-danger" style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="admin-stats">
        <div class="glass-card stat-card">
            <div class="label">Tổng số thuốc</div>
            <h2><?= number_format($stats['drugs']) ?></h2>
        </div>
        <div class="glass-card stat-card">
            <div class="label">Tổng số bệnh</div>
            <h2><?= number_format($stats['diseases']) ?></h2>
        </div>
        <div class="glass-card stat-card">
            <div class="label">Tổng số protein</div>
            <h2><?= number_format($stats['proteins']) ?></h2>
        </div>
        <div class="glass-card stat-card">
            <div class="label">Tổng lượt dự đoán</div>
            <h2><?= number_format($stats['predictions']) ?></h2>
        </div>
    </div>

    <div class="grid grid-2" style="grid-template-columns: 1fr 350px;">
        <div class="glass-card">
            <h3>Lượt dự đoán gần đây</h3>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Kiểu</th>
                        <th>Input</th>
                        <th>Kết quả</th>
                        <th>Thời gian</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($recent as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge" style="background: rgba(255,255,255,0.05);"><?= e((string) $row['query_type']) ?></span></td>
                            <td style="font-weight: 600;"><?= e((string) $row['input_text']) ?></td>
                            <td><span class="badge badge-drug">Top-<?= $row['top_k'] ?></span></td>
                            <td class="muted" style="font-size: 13px;"><?= date('H:i d/m', strtotime((string)$row['created_at'])) ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="glass-card">
            <h3>Hệ thống dữ liệu</h3>
            <p class="muted" style="margin-bottom: 20px;">Quản lý các thực thể và mối liên kết sinh học trong CSDL MySQL.</p>
            
            <div style="padding: 20px; background: rgba(59, 130, 246, 0.1); border-radius: 16px; border: 1px solid rgba(59,130,246,0.2);">
                <div class="label" style="color: #60a5fa;">Tổng liên kết hiện tại</div>
                <div style="font-size: 2rem; font-weight: 800; color: #fff;"><?= number_format($stats['links']) ?></div>
                <p class="muted" style="font-size: 12px; margin-top: 8px;">Dữ liệu này được sử dụng để xây dựng đồ thị cho mô hình HGT.</p>
            </div>

            <div style="margin-top: 24px; display: grid; gap: 12px;">
                <a href="admin_links.php" class="btn" style="width: 100%;">Quản lý liên kết ngay</a>
            </div>
        </div>
    </div>
</div>
</body>
</html>
