import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Space, Modal, message, Tag, Typography, Descriptions, Statistic, Row, Col, Alert, Form, Input, DatePicker, Tabs } from 'antd';
import { EyeOutlined, DeleteOutlined, ReloadOutlined, HistoryOutlined, SyncOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import axios from 'axios';
import dayjs from 'dayjs';
import { PaginationCookie } from '../utils/cookie';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  TitleComponent, TooltipComponent, GridComponent, DataZoomComponent,
  ToolboxComponent, LegendComponent
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { useParams, useNavigate } from 'react-router-dom';

// 注册 ECharts 必要的组件
echarts.use([
  TitleComponent, TooltipComponent, GridComponent, DataZoomComponent,
  ToolboxComponent, LegendComponent, LineChart, CanvasRenderer
]);

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface BacktestStatusRecord {
  id: number;
  name: string;
  description?: string;
  strategy_name?: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  instruments: string[];
  status: string;
  total_return: number;
  max_drawdown: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  performance_metrics?: {
    total_return?: number;
    max_drawdown?: number;
    sharpe_ratio?: number;
    volatility?: number;
    win_rate?: number;
    profit_factor?: number;
  };
}

interface BacktestHistoryRecord {
  id: number;
  status_id: number;
  start_date: string;
  end_date: string;
  initial_capital: number;
  instruments: string[];
  status: string;
  total_return: number;
  max_drawdown: number;
  created_at: string;
  completed_at?: string;
  operation_type: string;
  performance_metrics?: {
    total_return?: number;
    max_drawdown?: number;
    sharpe_ratio?: number;
    volatility?: number;
    win_rate?: number;
    profit_factor?: number;
  };
}

const BacktestHistoryDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [statusRecord, setStatusRecord] = useState<BacktestStatusRecord | null>(null);
  const [historyList, setHistoryList] = useState<BacktestHistoryRecord[]>([]);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedHistory, setSelectedHistory] = useState<BacktestHistoryRecord | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // 获取回测状态详情
  const fetchStatusDetail = async () => {
    if (!id) return;
    
    setLoading(true);
    try {
      const response = await axios.get(`/api/backtest-status/${id}`);
      if (response.data.status === 'success') {
        setStatusRecord(response.data.data);
      } else {
        message.error('获取回测状态失败');
      }
    } catch (error: any) {
      console.error('获取回测状态失败:', error);
      message.error('获取回测状态失败');
    } finally {
      setLoading(false);
    }
  };

  // 获取回测历史记录
  const fetchHistoryList = async () => {
    if (!id) return;
    
    setLoading(true);
    try {
      const response = await axios.get(`/api/backtest-status/${id}/history`);
      setHistoryList(response.data);
    } catch (error: any) {
      console.error('获取回测历史失败:', error);
      message.error('获取回测历史失败');
    } finally {
      setLoading(false);
    }
  };

  // 获取历史记录详情
  const fetchHistoryDetail = async (historyId: number) => {
    setDetailLoading(true);
    try {
      // 这里需要后端提供获取单个历史记录详情的API
      // 暂时使用历史记录列表中的数据
      const history = historyList.find(h => h.id === historyId);
      if (history) {
        setSelectedHistory(history);
        setDetailModalVisible(true);
      }
    } catch (error: any) {
      console.error('获取历史记录详情失败:', error);
      message.error('获取历史记录详情失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDeleteHistory = async (historyId: number) => {
    if (!id) return;
    
    Modal.confirm({
      title: '确认删除',
      content: '此操作将永久删除该历史记录，是否继续？',
      okText: '确认',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const response = await axios.delete(`/api/backtest-status/${id}/history/${historyId}`);
          if (response.data.status === 'success') {
            message.success('删除成功');
            if (selectedHistory && selectedHistory.id === historyId) {
              setDetailModalVisible(false);
              setSelectedHistory(null);
            }
            fetchHistoryList();
          } else {
            message.error(response.data.message || '删除失败');
          }
        } catch (error: any) {
          console.error('删除历史记录失败:', error);
          message.error('删除失败，请重试');
        }
      },
    });
  };

  // 历史记录表格列定义
  const historyColumns: ColumnsType<BacktestHistoryRecord> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '回测期间',
      key: 'period',
      render: (_, record) => (
        <div>
          <div>{dayjs(record.start_date).format('YYYY-MM-DD')}</div>
          <div style={{ color: '#666', fontSize: '12px' }}>
            至 {dayjs(record.end_date).format('YYYY-MM-DD')}
          </div>
        </div>
      ),
    },
    {
      title: '初始资金',
      dataIndex: 'initial_capital',
      key: 'initial_capital',
      render: (text) => `$${text.toLocaleString()}`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'completed' ? 'green' : status === 'running' ? 'blue' : 'red'}>
          {status === 'completed' ? '已完成' : status === 'running' ? '运行中' : '失败'}
        </Tag>
      ),
    },
    {
      title: '收益率',
      dataIndex: 'total_return',
      key: 'total_return',
      render: (text) => (
        <span style={{ color: text >= 0 ? '#52c41a' : '#ff4d4f' }}>
          {text >= 0 ? '+' : ''}{text.toFixed(2)}%
        </span>
      ),
    },
    {
      title: '最大回撤',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      render: (text) => (
        <span style={{ color: '#ff4d4f' }}>
          {text.toFixed(2)}%
        </span>
      ),
    },
    {
      title: '操作类型',
      dataIndex: 'operation_type',
      key: 'operation_type',
      render: (type) => (
        <Tag color={type === 'create' ? 'blue' : 'orange'}>
          {type === 'create' ? '创建' : type === 'update' ? '更新' : '重新运行'}
        </Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => dayjs(text).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => fetchHistoryDetail(record.id)}
            loading={detailLoading}
          >
            查看
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteHistory(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    fetchStatusDetail();
    fetchHistoryList();
  }, [id]);

  if (!statusRecord) {
    return (
      <div style={{ padding: '24px' }}>
        <Card loading={loading}>
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Text>加载中...</Text>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <Card style={{ marginBottom: '16px' }}>
        <Space>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/backtest/history')}
          >
            返回列表
          </Button>
          <Title level={3} style={{ margin: 0 }}>
            <HistoryOutlined /> {statusRecord.name} - 历史记录
          </Title>
        </Space>
      </Card>

      {/* 当前状态概览 */}
      <Card title="当前状态概览" style={{ marginBottom: '16px' }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="策略名称"
              value={statusRecord.strategy_name || '未知'}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="回测期间"
              value={`${dayjs(statusRecord.start_date).format('YYYY-MM-DD')} 至 ${dayjs(statusRecord.end_date).format('YYYY-MM-DD')}`}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="初始资金"
              value={`$${statusRecord.initial_capital.toLocaleString()}`}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="当前收益率"
              value={statusRecord.total_return}
              precision={2}
              suffix="%"
              valueStyle={{ color: statusRecord.total_return >= 0 ? '#52c41a' : '#ff4d4f' }}
            />
          </Col>
        </Row>
      </Card>

      {/* 历史记录表格 */}
      <Card
        title={
          <Space>
            <HistoryOutlined />
            <span>历史记录</span>
          </Space>
        }
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchHistoryList}
            loading={loading}
          >
            刷新
          </Button>
        }
      >
        <Table
          columns={historyColumns}
          dataSource={historyList}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: PaginationCookie.getPageSize(20),
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
            onChange: (page, pageSize) => {
              PaginationCookie.setCurrentPage(page);
              PaginationCookie.setPageSize(pageSize || 20);
            },
            onShowSizeChange: (current, size) => {
              PaginationCookie.setPageSize(size);
            }
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 历史记录详情模态框 */}
      <Modal
        title="历史记录详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedHistory && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="记录ID">{selectedHistory.id}</Descriptions.Item>
              <Descriptions.Item label="操作类型">
                <Tag color={selectedHistory.operation_type === 'create' ? 'blue' : 'orange'}>
                  {selectedHistory.operation_type === 'create' ? '创建' : '更新'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="回测期间" span={2}>
                {dayjs(selectedHistory.start_date).format('YYYY-MM-DD')} 至 {dayjs(selectedHistory.end_date).format('YYYY-MM-DD')}
              </Descriptions.Item>
              <Descriptions.Item label="初始资金">
                ${selectedHistory.initial_capital.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={selectedHistory.status === 'completed' ? 'green' : 'red'}>
                  {selectedHistory.status === 'completed' ? '已完成' : '失败'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="收益率">
                <span style={{ color: selectedHistory.total_return >= 0 ? '#52c41a' : '#ff4d4f' }}>
                  {selectedHistory.total_return >= 0 ? '+' : ''}{selectedHistory.total_return.toFixed(2)}%
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="最大回撤">
                <span style={{ color: '#ff4d4f' }}>
                  {selectedHistory.max_drawdown.toFixed(2)}%
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {dayjs(selectedHistory.created_at).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedHistory.completed_at ? dayjs(selectedHistory.completed_at).format('YYYY-MM-DD HH:mm:ss') : '未完成'}
              </Descriptions.Item>
            </Descriptions>

            {selectedHistory.performance_metrics && (
              <div style={{ marginTop: '16px' }}>
                <Title level={5}>性能指标</Title>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="累计收益"
                      value={(selectedHistory.total_return !== undefined
                              ? selectedHistory.total_return
                              : ((selectedHistory.performance_metrics.total_return || 0) * 100))}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="夏普比率"
                      value={selectedHistory.performance_metrics.sharpe_ratio || 0}
                      precision={3}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="波动率"
                      value={selectedHistory.performance_metrics.volatility || 0}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="胜率"
                      value={selectedHistory.performance_metrics.win_rate || 0}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                </Row>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default BacktestHistoryDetail;
