<?php

declare(strict_types=1);

return [
    'app_name' => 'Drug Disease AI Predictor',
    'base_url' => 'http://localhost/DoAnBaseFix/public',
    'db' => [
        'host' => '127.0.0.1',
        'port' => 3306,
        'dbname' => 'drug_disease_ai',
        'username' => 'root',
        'password' => '123456',
        'charset' => 'utf8mb4',
    ],
    'python_api' => [
        'base_url' => 'http://127.0.0.1:8000',
        'timeout' => 20,
    ],
    'session_name' => 'drug_disease_ai_session',
];
