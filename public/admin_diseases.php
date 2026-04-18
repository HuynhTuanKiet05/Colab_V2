<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? 'create';
    $sourceCode = trim($_POST['source_code'] ?? '');
    $name = trim($_POST['name'] ?? '');
    $description = trim($_POST['description'] ?? '');

    if ($action === 'create' && $sourceCode !== '' && $name !== '') {
        $stmt = db()->prepare('INSERT INTO diseases (source_code, name, description) VALUES (:source_code, :name, :description)');
        $stmt->execute([
            'source_code' => $sourceCode,
            'name' => $name,
            'description' => $description
        ]);
        flash('success', 'Đã thêm bệnh lý mới.');
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM diseases WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Đã xóa bệnh lý.');
    }

    redirect('admin_diseases.php');
}

$success = flash('success');
$rows = db()->query('SELECT * FROM diseases ORDER BY created_at DESC LIMIT 50')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Quản lý bệnh lý</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="brand">Admin <span style="font-weight: 300; opacity: 0.6;">Diseases Control</span></div>
        <div class="nav-links">
            <a class="btn btn-secondary" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);" href="admin.php">Quay lại Quản trị</a>
            <a class="btn btn-danger" style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="grid grid-2" style="grid-template-columns: 450px 1fr;">
        <div class="glass-card">
            <h3>Thêm bệnh lý mới</h3>
            <p class="muted" style="margin-bottom: 20px;">Dữ liệu bệnh lý sẽ được sử dụng trong việc dự đoán liên kết Thuốc-Bệnh.</p>
            
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div style="display: grid; gap: 16px;">
                    <div class="form-group">
                        <label class="label">Mã nguồn (Source Code)</label>
                        <input class="input" name="source_code" placeholder="Ví dụ: D102100" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Tên bệnh lý</label>
                        <input class="input" name="name" placeholder="Ví dụ: Lung Cancer" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Mô tả bệnh lý</label>
                        <textarea class="input" name="description" style="height: 150px; padding: 12px;" placeholder="Thông tin chi tiết về các triệu chứng hoặc mã phân loại..."></textarea>
                    </div>
                    <button class="btn" type="submit" style="width: 100%; margin-top: 10px;">Lưu thông tin bệnh lý</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <h3>Danh sách bệnh lý trong CSDL</h3>
            <div class="table-container" style="max-height: 600px; overflow-y: auto;">
                <table class="table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Mã nguồn</th>
                            <th>Tên bệnh lý</th>
                            <th style="text-align: right;">Hành động</th>
                        </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($rows)): ?>
                        <tr><td colspan="4" style="text-align: center; padding: 40px;" class="muted">Chưa có dữ liệu bệnh lý</td></tr>
                    <?php endif; ?>
                    <?php foreach ($rows as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge badge-disease"><?= e((string) $row['source_code']) ?></span></td>
                            <td style="font-weight: 600;"><?= e((string) $row['name']) ?></td>
                            <td style="text-align: right;">
                                <form method="post" onsubmit="return confirm('Bạn có chắc chắn muốn xóa bệnh lý này? Tất cả các lượt dự đoán liên quan sẽ mất thông tin thực thể này.');" style="display: inline;">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
                                    <button class="btn btn-danger" type="submit" style="padding: 6px 12px; font-size: 12px;">Xóa</button>
                                </form>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>
</body>
</html>
