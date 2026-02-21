import React, { useEffect, useMemo, useState } from 'react';
import { Card, Row, Col, Form, Input, DatePicker, Select, Button, Space, Tag, Table, message, Modal, Typography, Tooltip } from 'antd';
import { LineChartOutlined, PlayCircleOutlined, FileTextOutlined, StepForwardOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import OptimizedChart from '../components/OptimizedChart';
import axios from 'axios';
import { fetchStockList, Stock } from '../services/apiService';

const { RangePicker } = DatePicker;
const { Option } = Select;
const { Text } = Typography;

const MODEL_COLOR_MAP: Record<string, string> = {
  deepseek: '#3b7ddd',
  qwen: '#52c41a',
  kimi: '#faad14',
};

const MODEL_COLOR_PALETTE = ['#3b7ddd', '#52c41a', '#fa8c16', '#13c2c2', '#eb2f96'];

interface AiRun {
  id: number;
  name: string;
  symbol: string;
  models: string[];
  status: string;
  created_at: string;
  completed_at?: string;
}

interface AiRecord {
  id: number;
  run_id: number;
  model_type: string;
  timestamp: string;
  predicted_price: number;
  actual_price: number;
  action: string;
  position: number;
  pnl: number;
  cumulative_pnl: number;
  equity: number;
  trigger_reason?: string;
  prediction_reason?: string;
}

interface AiRunLog {
  id: number;
  timestamp: string;
  level: string;
  category: string;
  message: string;
  ai_input?: any;
  ai_output?: any;
  extra?: any;
}

interface AiRunLogResponse {
  total: number;
  items: AiRunLog[];
}

interface MetricsMap {
  [model: string]: {
    total_return: number;
    max_drawdown: number;
    sharpe_ratio: number;
  };
}

interface MetricsRow {
  model: string;
  total_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
}

interface EquityPoint {
  date: string;
  equity: number;
}

interface EquityCurvesMap {
  [model: string]: EquityPoint[];
}

interface PricePoint {
  date: string;
  close: number;
}

interface AiRunDetailResponse {
  status: string;
  message: string;
  run_id: number;
  metrics: MetricsMap;
  equity_curves: EquityCurvesMap;
  price_series: PricePoint[];
  name?: string;
  symbol?: string;
  data_source?: string;
  frequency?: string;
  models: string[];
  initial_capital?: number;
  config?: any;
}

interface AiPromptSettingItem {
  id: number;
  model_type: string;
  scene?: string;
  system_prompt: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

const DEFAULT_SYSTEM_PROMPT = [
  '你是一个专注于股票量化交易的金融大模型，充当价格预测引擎。',
  '',
  '你的任务是根据输入的K线数据、账户状态和最近成交记录，预测下一根K线的收盘价，并用一句话说明主要原因。',
  '',
  '要求：',
  '1）第一部分输出一个数字作为预测收盘价；',
  '2）第二部分用一句话总结主要原因，可以与数字在同一行，用逗号分隔；',
  '3）数字应为合理的价格水平，不为负数，尽量接近当前价格量级；',
  '4）可以保留2到6位小数；',
  '5）充分利用给出的周期、买入/卖出阈值、止损和止盈参数来判断趋势和风险；',
  '6）只使用输入的数据进行推断，不要编造外部信息或新闻；',
  '7）如果历史数据较少，也要给出尽可能稳健的预测，而不是报错。',
].join('\n');

const AiInvestment: React.FC = () => {
  const [form] = Form.useForm();
  const [runs, setRuns] = useState<AiRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [running, setRunning] = useState(false);
  const [selectedRun, setSelectedRun] = useState<AiRun | null>(null);
  const [records, setRecords] = useState<AiRecord[]>([]);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [metrics, setMetrics] = useState<MetricsMap>({});
  const [equityCurves, setEquityCurves] = useState<EquityCurvesMap>({});
  const [priceSeries, setPriceSeries] = useState<PricePoint[]>([]);
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loadingStocks, setLoadingStocks] = useState(false);
  const [promptModel, setPromptModel] = useState<string>('deepseek');
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [promptLoading, setPromptLoading] = useState(false);
  const [promptSaving, setPromptSaving] = useState(false);

  const [logModalVisible, setLogModalVisible] = useState(false);
  const [logRun, setLogRun] = useState<AiRun | null>(null);
  const [logs, setLogs] = useState<AiRunLog[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [logsPage, setLogsPage] = useState(1);
  const [logsPageSize, setLogsPageSize] = useState(50);
  const [logsTotal, setLogsTotal] = useState(0);
  const [logLevel, setLogLevel] = useState<string | undefined>();
  const [logCategory, setLogCategory] = useState<string | undefined>();
  const [logKeyword, setLogKeyword] = useState<string>('');
  const [logRecordId, setLogRecordId] = useState<number | undefined>();
  const [selectedRecord, setSelectedRecord] = useState<AiRecord | null>(null);
  const [estimatingCalls, setEstimatingCalls] = useState(false);
  const [callEstimateText, setCallEstimateText] = useState<string>('');
  const [resumingRunId, setResumingRunId] = useState<number | null>(null);

  const dataSourceValue = Form.useWatch('data_source', form) || 'database';

  const frequencyOptions = useMemo(
    () =>
      dataSourceValue === 'database'
        ? [{ value: '1d', label: '日线' }]
        : [
            { value: '1m', label: '1分钟' },
            { value: '5m', label: '5分钟' },
            { value: '15m', label: '15分钟' },
            { value: '30m', label: '30分钟' },
            { value: '60m', label: '60分钟' },
          ],
    [dataSourceValue],
  );

  const modelOrder = useMemo(() => {
    const set = new Set<string>();
    Object.keys(metrics || {}).forEach(m => set.add(m));
    Object.keys(equityCurves || {}).forEach(m => set.add(m));
    records.forEach(r => set.add(r.model_type));
    return Array.from(set);
  }, [metrics, equityCurves, records]);

  const getModelColor = (model: string) => {
    const index = modelOrder.indexOf(model);
    const baseColor =
      MODEL_COLOR_MAP[model] ||
      MODEL_COLOR_PALETTE[(index >= 0 ? index : 0) % MODEL_COLOR_PALETTE.length];
    return baseColor;
  };

  const fetchRuns = async () => {
    setLoadingRuns(true);
    try {
      const res = await axios.get<AiRun[]>('/api/ai-investment/runs');
      const list = res.data || [];
      setRuns(list);
      return list;
    } catch (error: any) {
      message.error(error?.message || '获取AI投资运行列表失败');
    } finally {
      setLoadingRuns(false);
    }
  };

  const fetchRecordsByRunId = async (runId: number) => {
    setRecordsLoading(true);
    try {
      const res = await axios.get<AiRecord[]>(`/api/ai-investment/run/${runId}/records`);
      const list = res.data || [];
      setRecords(list);
      return list;
    } catch (error: any) {
      message.error(error?.message || '获取AI投资预测明细失败');
    } finally {
      setRecordsLoading(false);
    }
  };

  const fetchRecords = async (run: AiRun) => {
    await fetchRecordsByRunId(run.id);
  };

  const fetchLogs = async (
    runId: number,
    page: number = logsPage,
    pageSize: number = logsPageSize,
    options?: { level?: string; category?: string; keyword?: string; recordId?: number },
  ) => {
    setLogsLoading(true);
    try {
      const params: any = {
        page,
        size: pageSize,
      };
      const levelValue = options?.level ?? logLevel;
      const categoryValue = options?.category ?? logCategory;
      const keywordValue = options?.keyword ?? logKeyword;
      const recordIdValue =
        options?.recordId !== undefined ? options.recordId : logRecordId;
      if (levelValue) {
        params.level = levelValue;
      }
      if (categoryValue) {
        params.category = categoryValue;
      }
      if (keywordValue) {
        params.keyword = keywordValue;
      }
      if (recordIdValue !== undefined) {
        params.record_id = recordIdValue;
      }
      const res = await axios.get<AiRunLogResponse>(
        `/api/ai-investment/run/${runId}/logs`,
        { params },
      );
      setLogs(res.data.items || []);
      setLogsTotal(res.data.total || 0);
      setLogsPage(page);
      setLogsPageSize(pageSize);
      if (options?.level !== undefined) {
        setLogLevel(options.level);
      }
      if (options?.category !== undefined) {
        setLogCategory(options.category);
      }
      if (options?.keyword !== undefined) {
        setLogKeyword(options.keyword || '');
      }
      if (options?.recordId !== undefined) {
        setLogRecordId(options.recordId);
      }
    } catch (error: any) {
      message.error(error?.message || '获取运行日志失败');
    } finally {
      setLogsLoading(false);
    }
  };

  const loadPromptSetting = async (modelType: string) => {
    setPromptLoading(true);
    try {
      const res = await axios.get<AiPromptSettingItem[]>('/api/ai-investment/prompt-settings', {
        params: { model_type: modelType },
      });
      const list = res.data || [];
      if (list.length > 0) {
        setSystemPrompt(list[0].system_prompt || '');
      } else {
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
      }
    } catch (error: any) {
      message.error(error?.message || '加载AI提示词失败');
      setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    } finally {
      setPromptLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
    loadPromptSetting(promptModel);
    const loadStocks = async () => {
      setLoadingStocks(true);
      try {
        const list = await fetchStockList();
        setStocks(list);
      } catch (error) {
      } finally {
        setLoadingStocks(false);
      }
    };
    loadStocks();
  }, []);

  useEffect(() => {
    const currentFrequency = form.getFieldValue('frequency');
    if (!currentFrequency) {
      if (dataSourceValue === 'database') {
        form.setFieldsValue({ frequency: '1d' });
      } else {
        form.setFieldsValue({ frequency: '1m' });
      }
    }
  }, [dataSourceValue, form]);

  useEffect(() => {
    loadPromptSetting(promptModel);
  }, [promptModel]);

  const handleRun = async () => {
    try {
      const values = await form.validateFields();
      const symbol: string = values.symbol;
      const dateRange = values.timeRange;
      const models: string[] = values.models || [];
      if (!symbol || !dateRange || models.length === 0) {
        message.warning('请填写股票代码、时间范围并选择至少一个模型');
        return;
      }

      const start = (dateRange[0] as dayjs.Dayjs).toDate().toISOString();
      const end = (dateRange[1] as dayjs.Dayjs).toDate().toISOString();

      const dataSource: string = values.data_source || 'database';
      const frequency: string =
        values.frequency || (dataSource === 'database' ? '1d' : '1m');

      const payload = {
        name: values.name || undefined,
        symbol,
        start_time: start,
        end_time: end,
        data_source: dataSource,
        frequency,
        models,
        initial_capital: values.initial_capital || 100000,
        buy_threshold: values.buy_threshold ?? 0.002,
        sell_threshold: values.sell_threshold ?? -0.002,
        stop_loss_pct: values.stop_loss_pct ?? 0.05,
        take_profit_pct: values.take_profit_pct ?? 0.1,
        window: values.window || 20,
        commission_rate: values.commission_rate ?? 0.0015,
        slippage_rate: values.slippage_rate ?? 0.001,
      };

      setRunning(true);
      const res = await axios.post('/api/ai-investment/run', payload);
      if (res.data && res.data.status === 'success') {
        message.success('AI实时投资回放完成');
        const data = res.data;
        setMetrics(data.metrics || {});
        setEquityCurves(data.equity_curves || {});
        setPriceSeries(data.price_series || []);
        const list = await fetchRuns();
        if (data.run_id) {
          await fetchRecordsByRunId(data.run_id);
          if (list && Array.isArray(list)) {
            const run = list.find(item => item.id === data.run_id) || null;
            setSelectedRun(run);
          }
        }
      } else {
        message.error(res.data?.message || 'AI投资回放失败');
      }
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      message.error(error?.message || 'AI投资回放执行失败');
    } finally {
      setRunning(false);
    }
  };

  const handleEstimateCalls = async () => {
    try {
      const values = await form.validateFields([
        'symbol',
        'timeRange',
        'data_source',
        'frequency',
        'models',
        'initial_capital',
        'buy_threshold',
        'sell_threshold',
        'stop_loss_pct',
        'take_profit_pct',
        'window',
      ]);
      const symbol: string = values.symbol;
      const dateRange = values.timeRange;
      const models: string[] = values.models || [];
      if (!symbol || !dateRange || models.length === 0) {
        message.warning('请先填写股票代码、时间范围并选择至少一个模型');
        return;
      }
      const start = (dateRange[0] as dayjs.Dayjs).toDate().toISOString();
      const end = (dateRange[1] as dayjs.Dayjs).toDate().toISOString();
      const dataSource: string = values.data_source || 'database';
      const frequency: string =
        values.frequency || (dataSource === 'database' ? '1d' : '1m');
      const payload = {
        name: values.name || undefined,
        symbol,
        start_time: start,
        end_time: end,
        data_source: dataSource,
        frequency,
        models,
        initial_capital: values.initial_capital || 100000,
        buy_threshold: values.buy_threshold ?? 0.002,
        sell_threshold: values.sell_threshold ?? -0.002,
        stop_loss_pct: values.stop_loss_pct ?? 0.05,
        take_profit_pct: values.take_profit_pct ?? 0.1,
        window: values.window || 20,
        commission_rate: values.commission_rate ?? 0.0015,
        slippage_rate: values.slippage_rate ?? 0.001,
      };
      setEstimatingCalls(true);
      const res = await axios.post('/api/ai-investment/estimate-calls', payload);
      const data = res.data;
      const formulaText = `${data.formula} = ${data.per_model_calls}`;
      const totalText = `当前选择 ${data.model_count} 个模型，理论最大AI调用次数约为 ${data.total_calls} 次。`;
      setCallEstimateText(`调用次数估算公式：${formulaText}。${totalText}`);
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      message.error(error?.message || '预估AI调用次数失败');
    } finally {
      setEstimatingCalls(false);
    }
  };

  const handleSavePrompt = async () => {
    if (!promptModel) {
      message.warning('请选择需要配置提示词的模型');
      return;
    }
    if (!systemPrompt || !systemPrompt.trim()) {
      message.warning('请输入系统提示词内容');
      return;
    }
    setPromptSaving(true);
    try {
      await axios.post<AiPromptSettingItem>('/api/ai-investment/prompt-settings', {
        model_type: promptModel,
        scene: 'ai_investment',
        system_prompt: systemPrompt,
      });
      message.success('提示词已保存');
    } catch (error: any) {
      message.error(error?.message || '保存提示词失败');
    } finally {
      setPromptSaving(false);
    }
  };

  const handleRowClick = (run: AiRun) => {
    setSelectedRun(run);
    fetchRecords(run);
  };

  const handleResumeRun = async (run: AiRun) => {
    let estimateData: any | null = null;
    try {
      const estimateRes = await axios.post(
        `/api/ai-investment/run/${run.id}/estimate-calls-resume`,
        {},
      );
      estimateData = estimateRes.data;
    } catch (e) {
    }

    Modal.confirm({
      title: '确认继续运行？',
      content: (
        <div>
          {estimateData ? (
            estimateData.total_calls > 0 ? (
              <>
                <p>
                  按当前配置，理论最大 AI 调用次数约为{' '}
                  <Text strong>{estimateData.total_calls}</Text> 次。
                </p>
                <p>
                  每个模型约 <Text strong>{estimateData.per_model_calls}</Text> 次，当前选择{' '}
                  <Text strong>{estimateData.model_count}</Text> 个模型。
                </p>
              </>
            ) : (
              <p>根据当前数据，续跑已无剩余预测步数。</p>
            )
          ) : (
            <p>暂时无法预估本次继续运行的 AI 调用次数。</p>
          )}
          <p>确定要继续运行该任务吗？</p>
        </div>
      ),
      onOk: async () => {
        setResumingRunId(run.id);
        try {
          const res = await axios.post<AiRunDetailResponse>(
            `/api/ai-investment/run/${run.id}/resume`,
            {
              name: `${run.name}-续跑`,
            },
          );
          const data = res.data;
          if (data && data.status === 'success') {
            message.success('续跑任务已完成');

            const detailRes = await axios.get<AiRunDetailResponse>(
              `/api/ai-investment/run/${run.id}`,
            );
            const detail = detailRes.data;

            setSelectedRun({
              ...run,
              name: detail.name || run.name,
            });
            setMetrics(detail.metrics || {});
            setEquityCurves(detail.equity_curves || {});
            setPriceSeries(detail.price_series || []);

            await fetchRecords(run);
            const list = await fetchRuns();
            if (list && Array.isArray(list)) {
              setRuns(list);
            }
          } else {
            message.error(data?.message || '续跑任务执行失败');
          }
        } catch (error: any) {
          message.error(error?.message || '续跑任务请求失败');
        } finally {
          setResumingRunId(null);
        }
      },
    });
  };

  const handleLoadRun = async (run: AiRun) => {
    try {
      const res = await axios.get(`/api/ai-investment/run/${run.id}`);
      const data = res.data;
      if (data && data.status === 'success') {
        setSelectedRun(run);
        setMetrics(data.metrics || {});
        setEquityCurves(data.equity_curves || {});
        setPriceSeries(data.price_series || []);

        const config = data.config || {};
        const startTime = config.start_time as string | undefined;

        const recordsList = await fetchRecordsByRunId(run.id);

        let endTime = config.end_time as string | undefined;
        if (recordsList && recordsList.length > 0) {
          const latestRecordTime = recordsList[recordsList.length - 1].timestamp;
          if (!endTime || dayjs(endTime).isBefore(latestRecordTime)) {
            endTime = latestRecordTime;
          }
        }
        if (!endTime) {
          const priceSeriesList = data.price_series || [];
          if (priceSeriesList.length > 0) {
            endTime = priceSeriesList[priceSeriesList.length - 1].date;
          }
        }
        if (!endTime && run.completed_at) {
          endTime = run.completed_at;
        }
        const timeRange =
          startTime && endTime ? [dayjs(startTime), dayjs(endTime)] : undefined;

        const formValues: any = {
          name: data.name || run.name,
          symbol: data.symbol || run.symbol,
          data_source: data.data_source,
          frequency: data.frequency,
          models:
            Array.isArray(data.models) && data.models.length > 0
              ? data.models
              : run.models,
          initial_capital: data.initial_capital,
          buy_threshold: config.buy_threshold,
          sell_threshold: config.sell_threshold,
          stop_loss_pct: config.stop_loss_pct,
          take_profit_pct: config.take_profit_pct,
          window: config.window,
          commission_rate: config.commission_rate,
          slippage_rate: config.slippage_rate,
        };

        if (timeRange) {
          formValues.timeRange = timeRange;
        }

        Object.keys(formValues).forEach(key => {
          const value = formValues[key];
          if (value === undefined || value === null) {
            delete formValues[key];
          }
        });

        if (Object.keys(formValues).length > 0) {
          form.setFieldsValue(formValues);
        }
        message.success('已载入历史运行结果');
      } else {
        message.error(data?.message || '载入历史运行结果失败');
      }
    } catch (error: any) {
      message.error(error?.message || '载入历史运行结果失败');
    }
  };

  const handleDeleteRun = async (run: AiRun) => {
    Modal.confirm({
      title: '确认删除该历史运行？',
      content: '删除后将无法恢复该运行记录及相关明细，是否继续？',
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const res = await axios.delete(`/api/ai-investment/run/${run.id}`);
          const data = res.data;
          if (data && data.status === 'success') {
            message.success('删除成功');
            const list = await fetchRuns();
            if (selectedRun && selectedRun.id === run.id) {
              setSelectedRun(null);
              setRecords([]);
              setMetrics({});
              setEquityCurves({});
              setPriceSeries([]);
            }
            if (logRun && logRun.id === run.id) {
              setLogModalVisible(false);
              setLogRun(null);
              setLogs([]);
            }
            if (list && Array.isArray(list)) {
              setRuns(list);
            }
          } else {
            message.error(data?.message || '删除失败');
          }
        } catch (error: any) {
          message.error(error?.message || '删除运行记录失败');
        }
      },
    });
  };

  const handleViewRunLogs = (run: AiRun) => {
    setLogRun(run);
    setSelectedRecord(null);
    setLogRecordId(undefined);
    setLogLevel(undefined);
    setLogCategory(undefined);
    setLogKeyword('');
    setLogModalVisible(true);
    fetchLogs(run.id, 1, logsPageSize, {
      level: undefined,
      category: undefined,
      keyword: '',
      recordId: undefined,
    });
  };

  const handleViewRecordLogs = (record: AiRecord) => {
    const relatedRun = runs.find(r => r.id === record.run_id) || selectedRun;
    if (!relatedRun) {
      message.warning('请先在左侧选择对应的运行记录');
      return;
    }
    setLogRun(relatedRun);
    setSelectedRecord(record);
    setLogLevel(undefined);
    setLogCategory('ai_call');
    setLogKeyword('');
    setLogRecordId(record.id);
    setLogModalVisible(true);
    fetchLogs(record.run_id, 1, logsPageSize, {
      category: 'ai_call',
      recordId: record.id,
    });
  };

  const predictionAiCallLog = useMemo(() => {
    if (!selectedRecord) {
      return undefined;
    }
    return logs.find(log => log.category === 'ai_call');
  }, [logs, selectedRecord]);

  const runColumns: ColumnsType<AiRun> = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      ellipsis: true,
    },
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: '模型',
      dataIndex: 'models',
      key: 'models',
      width: 140,
      render: (models: string[]) => (
        <Space size="small">
          {(models || []).map(m => (
            <Tag key={m}>{m}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={status === 'completed' ? 'green' : 'blue'}>{status}</Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (v: string) => (v ? dayjs(v).format('YYYY-MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '完成时间',
      dataIndex: 'completed_at',
      key: 'completed_at',
      width: 180,
      render: (v?: string) => (v ? dayjs(v).format('YYYY-MM-DD HH:mm:ss') : '-'),
    },
    {
      title: '操作',
      key: 'action',
      width: 220,
      render: (_: any, record: AiRun) => (
        <Space size="small">
          <Button
            size="small"
            icon={<FileTextOutlined />}
            onClick={e => {
              e.stopPropagation();
              handleViewRunLogs(record);
            }}
          >
            查看日志
          </Button>
          <Button
            size="small"
            icon={<StepForwardOutlined />}
            loading={resumingRunId === record.id}
            onClick={e => {
              e.stopPropagation();
              handleResumeRun(record);
            }}
          >
            继续运行
          </Button>
          <Button
            size="small"
            icon={<PlayCircleOutlined />}
            onClick={e => {
              e.stopPropagation();
              handleLoadRun(record);
            }}
          >
            载入结果
          </Button>
          <Button
            size="small"
            danger
            onClick={e => {
              e.stopPropagation();
              handleDeleteRun(record);
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const recordColumns: ColumnsType<AiRecord> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (v: string) => (v ? dayjs(v).format('YYYY-MM-DD HH:mm') : '-'),
    },
    {
      title: '模型',
      dataIndex: 'model_type',
      key: 'model_type',
    },
    {
      title: '预测价格',
      dataIndex: 'predicted_price',
      key: 'predicted_price',
      render: (v: number) => v.toFixed(4),
    },
    {
      title: '实际价格',
      dataIndex: 'actual_price',
      key: 'actual_price',
      render: (v: number) => v.toFixed(4),
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      render: (text: string) => {
        let color: string = 'default';
        let label = text;
        if (text === 'BUY') {
          color = 'red';
          label = '买入';
        } else if (text === 'SELL') {
          color = 'green';
          label = '卖出';
        } else if (text === 'HOLD') {
          color = 'default';
          label = '观望';
        }
        return <Tag color={color}>{label}</Tag>;
      },
    },
    {
      title: '触发原因',
      dataIndex: 'trigger_reason',
      key: 'trigger_reason',
      width: 160,
      ellipsis: { showTitle: false },
      render: (text?: string) =>
        text ? (
          <Tooltip title={text} mouseEnterDelay={0} mouseLeaveDelay={0}>
            <span>{text}</span>
          </Tooltip>
        ) : (
          '—'
        ),
    },
    {
      title: '预测原因',
      dataIndex: 'prediction_reason',
      key: 'prediction_reason',
      width: 160,
      ellipsis: { showTitle: false },
      render: (text?: string) =>
        text ? (
          <Tooltip title={text} mouseEnterDelay={0} mouseLeaveDelay={0}>
            <span>{text}</span>
          </Tooltip>
        ) : (
          '—'
        ),
    },
    {
      title: '持仓',
      dataIndex: 'position',
      key: 'position',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '单笔收益',
      dataIndex: 'pnl',
      key: 'pnl',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '累计收益',
      dataIndex: 'cumulative_pnl',
      key: 'cumulative_pnl',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '账户权益',
      dataIndex: 'equity',
      key: 'equity',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: AiRecord) => (
        <Button
          size="small"
          icon={<FileTextOutlined />}
          onClick={() => handleViewRecordLogs(record)}
        >
          查看日志
        </Button>
      ),
    },
  ];

  const equityChartOption = useMemo(() => {
    const series: any[] = [];
    const dates: string[] = [];
    Object.keys(equityCurves || {}).forEach(model => {
      const points = equityCurves[model] || [];
      const modelDates = points.map(p => dayjs(p.date).format('YYYY-MM-DD HH:mm'));
      if (modelDates.length > dates.length) {
        dates.splice(0, dates.length, ...modelDates);
      }
    });
    Object.keys(equityCurves || {}).forEach(model => {
      const points = equityCurves[model] || [];
      const values = points.map(p => p.equity);
      const color = getModelColor(model);
      series.push({
        type: 'line',
        name: model,
        data: values,
        smooth: true,
        lineStyle: {
          color,
        },
        itemStyle: {
          color,
        },
      });
    });
    return {
      tooltip: { trigger: 'axis' },
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 80 },
      xAxis: {
        type: 'category',
        data: dates,
      },
      yAxis: {
        type: 'value',
        scale: true,
        name: '账户权益',
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          show: true,
          type: 'slider',
          bottom: 20,
          start: 0,
          end: 100,
        },
      ],
      series,
    };
  }, [equityCurves]);

  const priceChartOption = useMemo(() => {
    const dateFormat = 'YYYY-MM-DD HH:mm';
    const baseDates: string[] = [];

    if (priceSeries.length > 0) {
      priceSeries.forEach(p => {
        baseDates.push(dayjs(p.date).format(dateFormat));
      });
    } else if (records.length > 0) {
      const dateSet = new Set<string>();
      records.forEach(r => {
        dateSet.add(dayjs(r.timestamp).format(dateFormat));
      });
      Array.from(dateSet)
        .sort()
        .forEach(d => baseDates.push(d));
    }

    if (baseDates.length === 0) {
      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross',
            label: {
              backgroundColor: '#555',
            },
          },
        },
        legend: {
          top: 0,
          data: [],
        },
        grid: { left: 50, right: 30, top: 40, bottom: 80 },
        xAxis: {
          type: 'category',
          data: [],
        },
        yAxis: {
          type: 'value',
          scale: true,
          name: '价格',
        },
        dataZoom: [
          {
            type: 'inside',
            start: 0,
            end: 100,
          },
          {
            show: true,
            type: 'slider',
            bottom: 20,
            start: 0,
            end: 100,
          },
        ],
        series: [],
      };
    }

    const actualMap = new Map<string, number>();
    priceSeries.forEach(p => {
      const key = dayjs(p.date).format(dateFormat);
      actualMap.set(key, p.close);
    });
    if (actualMap.size === 0 && records.length > 0) {
      records.forEach(r => {
        const key = dayjs(r.timestamp).format(dateFormat);
        if (!actualMap.has(key)) {
          actualMap.set(key, r.actual_price);
        }
      });
    }
    const actualValues = baseDates.map(d => actualMap.get(d) ?? null);

    const predictedSeries: any[] = [];
    modelOrder.forEach(model => {
      const modelRecords = records.filter(r => r.model_type === model);
      if (modelRecords.length === 0) {
        return;
      }
      const predMap = new Map<string, number>();
      modelRecords.forEach(r => {
        const key = dayjs(r.timestamp).format(dateFormat);
        predMap.set(key, r.predicted_price);
      });
      const values = baseDates.map(d => predMap.get(d) ?? null);
      const color = getModelColor(model);
      predictedSeries.push({
        type: 'line',
        name: `${model}预测`,
        data: values,
        smooth: true,
        lineStyle: {
          color,
        },
        itemStyle: {
          color,
        },
      });
    });

    const dateIndexMap = new Map<string, number>();
    baseDates.forEach((d, index) => {
      dateIndexMap.set(d, index);
    });

    const buyPoints: { index: number; price: number }[] = [];
    const sellPoints: { index: number; price: number }[] = [];

    records.forEach(r => {
      const key = dayjs(r.timestamp).format(dateFormat);
      const index = dateIndexMap.get(key);
      if (index === undefined) {
        return;
      }
      const price = r.actual_price ?? actualMap.get(key) ?? null;
      if (price == null) {
        return;
      }
      if (r.action === 'BUY') {
        buyPoints.push({ index, price });
      } else if (r.action === 'SELL') {
        sellPoints.push({ index, price });
      }
    });

    const buyCirclePoints = buyPoints.map(p => ({
      name: '买入点',
      coord: [baseDates[p.index], p.price],
      value: p.price,
      itemStyle: {
        color: '#f64034',
      },
      symbol: 'circle',
      symbolSize: 8,
      label: {
        show: false,
      },
    }));

    const sellCirclePoints = sellPoints.map(p => ({
      name: '卖出点',
      coord: [baseDates[p.index], p.price],
      value: p.price,
      itemStyle: {
        color: '#00b46a',
      },
      symbol: 'circle',
      symbolSize: 8,
      label: {
        show: false,
      },
    }));

    const buyLabelPoints = buyPoints.map(p => ({
      name: '买入',
      coord: [baseDates[p.index], p.price * 1.02],
      itemStyle: {
        color: '#f64034',
      },
      symbol: 'pin',
      symbolSize: 22,
      label: {
        show: true,
        formatter: 'B',
        color: '#ffffff',
        position: 'inside',
        fontSize: 12,
        fontWeight: 'bold',
      },
    }));

    const sellLabelPoints = sellPoints.map(p => ({
      name: '卖出',
      coord: [baseDates[p.index], p.price * 0.98],
      itemStyle: {
        color: '#00b46a',
      },
      symbol: 'pin',
      symbolSize: 22,
      label: {
        show: true,
        formatter: 'S',
        color: '#ffffff',
        position: 'inside',
        fontSize: 12,
        fontWeight: 'bold',
      },
    }));

    const markPointData = [
      ...buyCirclePoints,
      ...sellCirclePoints,
      ...buyLabelPoints,
      ...sellLabelPoints,
    ];

    const series: any[] = [
      {
        type: 'line',
        name: '实际收盘价',
        data: actualValues,
        smooth: true,
        lineStyle: {
          color: '#666666',
        },
        itemStyle: {
          color: '#666666',
        },
        markPoint: {
          data: markPointData,
        },
      },
      ...predictedSeries,
    ];

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: '#555',
          },
        },
      },
      legend: {
        top: 0,
        data: series.map(s => s.name),
      },
      grid: { left: 50, right: 30, top: 40, bottom: 80 },
      xAxis: {
        type: 'category',
        data: baseDates,
      },
      yAxis: {
        type: 'value',
        scale: true,
        name: '价格',
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          show: true,
          type: 'slider',
          bottom: 20,
          start: 0,
          end: 100,
        },
      ],
      series,
    };
  }, [priceSeries, records, modelOrder]);

  const metricsData = useMemo<MetricsRow[]>(() => {
    if (!metrics) {
      return [];
    }
    return Object.keys(metrics).map(model => {
      const m = metrics[model];
      return {
        model,
        total_return: m?.total_return ?? 0,
        max_drawdown: m?.max_drawdown ?? 0,
        sharpe_ratio: m?.sharpe_ratio ?? 0,
      };
    });
  }, [metrics]);

  const metricsColumns: ColumnsType<MetricsRow> = [
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
    },
    {
      title: '总收益率',
      dataIndex: 'total_return',
      key: 'total_return',
      render: (v: number) => `${(v * 100).toFixed(2)} %`,
    },
    {
      title: '最大回撤',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      render: (v: number) => `${(v * 100).toFixed(2)} %`,
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      render: (v: number) => v.toFixed(2),
    },
  ];

  return (
    <div>
      <Row gutter={16}>
        <Col span={16}>
          <Card
            title={
              <Space>
                <PlayCircleOutlined />
                AI实时投资跑数配置
              </Space>
            }
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                data_source: 'database',
                frequency: '1d',
                models: ['deepseek'],
                initial_capital: 100000,
                buy_threshold: 0.002,
                sell_threshold: -0.002,
                stop_loss_pct: 0.05,
                take_profit_pct: 0.1,
                window: 20,
                commission_rate: 0.0015,
                slippage_rate: 0.001,
              }}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item name="name" label="运行名称">
                    <Input placeholder="可选，例如：AAPL-日内回放" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  {dataSourceValue === 'database' ? (
                    <Form.Item
                      name="symbol"
                      label="股票代码"
                      rules={[{ required: true, message: '请选择股票代码' }]}
                    >
                      <Select
                        showSearch
                        placeholder="从市场数据管理中选择股票"
                        loading={loadingStocks}
                        optionFilterProp="children"
                        filterOption={(input, option) =>
                          String(option?.children ?? '')
                            .toLowerCase()
                            .includes(input.toLowerCase())
                        }
                      >
                        {stocks.map(stock => (
                          <Option key={stock.id} value={stock.symbol}>
                            {stock.symbol}（{stock.name}）
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                  ) : (
                    <Form.Item
                      name="symbol"
                      label="股票代码"
                      rules={[{ required: true, message: '请输入股票代码' }]}
                    >
                      <Input placeholder="例如：HK.00700" />
                    </Form.Item>
                  )}
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="timeRange"
                    label="时间范围"
                    rules={[{ required: true, message: '请选择时间范围' }]}
                  >
                    <RangePicker
                      showTime
                      style={{ width: '100%' }}
                      presets={[
                        {
                          label: '最近3天',
                          value: [dayjs().subtract(3, 'day'), dayjs()],
                        },
                        {
                          label: '最近一周',
                          value: [dayjs().subtract(7, 'day'), dayjs()],
                        },
                        {
                          label: '最近两周',
                          value: [dayjs().subtract(14, 'day'), dayjs()],
                        },
                        {
                          label: '最近3周',
                          value: [dayjs().subtract(21, 'day'), dayjs()],
                        },
                        {
                          label: '最近一月',
                          value: [dayjs().subtract(1, 'month'), dayjs()],
                        },
                        {
                          label: '最近3月',
                          value: [dayjs().subtract(3, 'month'), dayjs()],
                        },
                        {
                          label: '最近半年',
                          value: [dayjs().subtract(6, 'month'), dayjs()],
                        },
                        {
                          label: '最近一年',
                          value: [dayjs().subtract(1, 'year'), dayjs()],
                        },
                        {
                          label: '最近两年',
                          value: [dayjs().subtract(2, 'year'), dayjs()],
                        },
                        {
                          label: '最近3年',
                          value: [dayjs().subtract(3, 'year'), dayjs()],
                        },
                        {
                          label: '最近5年',
                          value: [dayjs().subtract(5, 'year'), dayjs()],
                        },
                      ]}
                      disabledDate={current => current && current > dayjs().endOf('day')}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="models"
                    label="AI模型"
                    rules={[{ required: true, message: '请选择至少一个模型' }]}
                  >
                    <Select
                      mode="multiple"
                      placeholder="支持多选，用于模型对比或加权"
                    >
                      <Option value="qwen">千问</Option>
                      <Option value="kimi">Kimi</Option>
                      <Option value="deepseek">DeepSeek</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item name="data_source" label="数据源">
                    <Select>
                      <Option value="database">市场数据管理(日线)</Option>
                      <Option value="futu">富途 OpenAPI 实时</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="frequency" label="数据频率">
                    <Select>
                      {frequencyOptions.map(item => (
                        <Option key={item.value} value={item.value}>
                          {item.label}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="initial_capital" label="初始资金">
                    <Input type="number" min={0} />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={6}>
                  <Form.Item name="buy_threshold" label="买入阈值(预测涨幅)">
                    <Input type="number" step="0.001" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="sell_threshold" label="卖出阈值(预测跌幅)">
                    <Input type="number" step="0.001" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="stop_loss_pct" label="止损比例">
                    <Input type="number" step="0.01" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="take_profit_pct" label="止盈比例">
                    <Input type="number" step="0.01" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={6}>
                  <Form.Item name="commission_rate" label="手续费率">
                    <Input type="number" step="0.0001" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item name="slippage_rate" label="滑点比例">
                    <Input type="number" step="0.0001" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={6}>
                  <Form.Item name="window" label="模型窗口长度">
                    <Input type="number" min={2} />
                  </Form.Item>
                </Col>
              </Row>
              <Row justify="space-between" align="middle">
                <Col>
                  <Space direction="vertical" size={4}>
                    <Button
                      size="small"
                      loading={estimatingCalls}
                      onClick={handleEstimateCalls}
                    >
                      预估AI调用次数
                    </Button>
                    {callEstimateText && (
                      <Text type="danger" style={{ maxWidth: 480 }}>
                        {callEstimateText}
                      </Text>
                    )}
                  </Space>
                </Col>
                <Col>
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    loading={running}
                    onClick={handleRun}
                  >
                    启动AI跑数
                  </Button>
                </Col>
              </Row>
            </Form>
          </Card>
        </Col>
        <Col span={8}>
          <Card
            title={
              <Space>
                <LineChartOutlined />
                历史运行
              </Space>
            }
          >
            <Table
              rowKey="id"
              loading={loadingRuns}
              dataSource={runs}
              columns={runColumns}
              size="small"
              scroll={{ x: true }}
              pagination={{ pageSize: 5 }}
              onRow={record => ({
                onClick: () => handleRowClick(record),
              })}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="价格与模型预测（含买卖点）">
            <OptimizedChart option={priceChartOption} style={{ height: 540 }} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="账户权益曲线">
            <OptimizedChart option={equityChartOption} style={{ height: 540 }} />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="预测明细">
            <Table
              rowKey={(r, index) => `${r.model_type}-${r.timestamp}-${index}`}
              loading={recordsLoading}
              dataSource={records}
              columns={recordColumns}
              size="small"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 1200 }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="模型绩效对比">
            <Table
              rowKey="model"
              dataSource={metricsData}
              columns={metricsColumns}
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="AI提示词设置">
            <Row gutter={16}>
              <Col span={8}>
                <Space>
                  <span>选择模型</span>
                  <Select
                    style={{ minWidth: 160 }}
                    value={promptModel}
                    onChange={value => setPromptModel(value)}
                  >
                    <Option value="deepseek">DeepSeek</Option>
                    <Option value="qwen">千问</Option>
                    <Option value="kimi">Kimi</Option>
                  </Select>
                </Space>
              </Col>
            </Row>
            <Row style={{ marginTop: 16 }}>
              <Col span={24}>
                <Input.TextArea
                  value={systemPrompt}
                  onChange={e => setSystemPrompt(e.target.value)}
                  autoSize={{ minRows: 4, maxRows: 20 }}
                  style={{ whiteSpace: 'pre-wrap' }}
                  placeholder="请输入系统提示词，将作为大模型的系统角色描述"
                />
              </Col>
            </Row>
            <Row justify="end" style={{ marginTop: 16 }}>
              <Col>
                <Space>
                  <Button onClick={() => loadPromptSetting(promptModel)} loading={promptLoading}>
                    重新加载
                  </Button>
                  <Button
                    type="primary"
                    loading={promptSaving}
                    onClick={handleSavePrompt}
                  >
                    保存提示词
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Modal
        open={logModalVisible}
        title={
          logRun
            ? `运行日志 - ${logRun.name} (#${logRun.id}) · 共${logsTotal}条`
            : `运行日志 · 共${logsTotal}条`
        }
        onCancel={() => setLogModalVisible(false)}
        footer={null}
        width={960}
        >
        {selectedRecord && (
          <Card size="small" style={{ marginBottom: 16 }}>
            <Row gutter={16} style={{ marginBottom: 8 }}>
              <Col span={8}>
                <Text strong>时间</Text>
                <div>{dayjs(selectedRecord.timestamp).format('YYYY-MM-DD HH:mm')}</div>
              </Col>
              <Col span={8}>
                <Text strong>模型</Text>
                <div>{selectedRecord.model_type}</div>
              </Col>
              <Col span={8}>
                <Text strong>操作</Text>
                <div>{selectedRecord.action}</div>
              </Col>
            </Row>
            <Row gutter={16} style={{ marginBottom: 12 }}>
              <Col span={8}>
                <Text strong>预测价格</Text>
                <div>{selectedRecord.predicted_price.toFixed(4)}</div>
              </Col>
              <Col span={8}>
                <Text strong>实际价格</Text>
                <div>{selectedRecord.actual_price.toFixed(4)}</div>
              </Col>
              <Col span={8}>
                <Text strong>账户权益</Text>
                <div>{selectedRecord.equity.toFixed(2)}</div>
              </Col>
            </Row>
            {selectedRecord.prediction_reason && (
              <Row gutter={16} style={{ marginBottom: 12 }}>
                <Col span={24}>
                  <Text strong>预测原因</Text>
                  <div>{selectedRecord.prediction_reason}</div>
                </Col>
              </Row>
            )}
            {predictionAiCallLog && (
              <>
                <Row gutter={16} style={{ marginBottom: 8 }}>
                  <Col span={24}>
                    <Text strong>模型输入</Text>
                  </Col>
                </Row>
                <Row gutter={16} style={{ marginBottom: 12 }}>
                  <Col span={24}>
                    <div style={{ fontSize: 12, color: '#888' }}>System Prompt</div>
                    <div
                      style={{
                        border: '1px solid #f0f0f0',
                        borderRadius: 4,
                        padding: 8,
                        maxHeight: 120,
                        overflow: 'auto',
                        marginTop: 4,
                        whiteSpace: 'pre-wrap',
                        background: '#fafafa',
                      }}
                    >
                      {predictionAiCallLog.ai_input?.system_prompt || '无'}
                    </div>
                  </Col>
                </Row>
                <Row gutter={16} style={{ marginBottom: 12 }}>
                  <Col span={24}>
                    <div style={{ fontSize: 12, color: '#888' }}>Prompt</div>
                    <div
                      style={{
                        border: '1px solid #f0f0f0',
                        borderRadius: 4,
                        padding: 8,
                        maxHeight: 120,
                        overflow: 'auto',
                        marginTop: 4,
                        whiteSpace: 'pre-wrap',
                        background: '#fafafa',
                      }}
                    >
                      {predictionAiCallLog.ai_input?.prompt || '无'}
                    </div>
                  </Col>
                </Row>
                <Row gutter={16} style={{ marginBottom: 8 }}>
                  <Col span={24}>
                    <Text strong>模型输出</Text>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col span={8}>
                    <Text strong>解析后的数值</Text>
                    <div>
                      {predictionAiCallLog.ai_output &&
                      predictionAiCallLog.ai_output.value !== undefined
                        ? predictionAiCallLog.ai_output.value
                        : '无'}
                    </div>
                  </Col>
                  <Col span={16}>
                    <div style={{ fontSize: 12, color: '#888' }}>原始内容</div>
                    <div
                      style={{
                        border: '1px solid #f0f0f0',
                        borderRadius: 4,
                        padding: 8,
                        maxHeight: 120,
                        overflow: 'auto',
                        marginTop: 4,
                        whiteSpace: 'pre-wrap',
                        background: '#fafafa',
                      }}
                    >
                      {predictionAiCallLog.ai_output?.content || '无'}
                    </div>
                  </Col>
                </Row>
              </>
            )}
          </Card>
        )}
        <Space style={{ marginBottom: 8 }}>
          <Select
            allowClear
            placeholder="级别"
            style={{ width: 120 }}
            value={logLevel}
            onChange={value => {
              if (!logRun) {
                return;
              }
              fetchLogs(logRun.id, 1, logsPageSize, {
                level: value || undefined,
              });
            }}
          >
            <Option value="DEBUG">DEBUG</Option>
            <Option value="INFO">INFO</Option>
            <Option value="WARN">WARN</Option>
            <Option value="ERROR">ERROR</Option>
          </Select>
          <Select
            allowClear
            placeholder="类别"
            style={{ width: 140 }}
            value={logCategory}
            onChange={value => {
              if (!logRun) {
                return;
              }
              fetchLogs(logRun.id, 1, logsPageSize, {
                category: value || undefined,
              });
            }}
          >
            <Option value="run">运行</Option>
            <Option value="ai_call">AI调用</Option>
            <Option value="trade">买卖点</Option>
            <Option value="system">系统</Option>
          </Select>
          <Input.Search
            allowClear
            placeholder="按关键字搜索"
            style={{ width: 260 }}
            value={logKeyword}
            onChange={e => setLogKeyword(e.target.value)}
            onSearch={value => {
              if (!logRun) {
                return;
              }
              fetchLogs(logRun.id, 1, logsPageSize, {
                keyword: value,
              });
            }}
          />
          <span>共 {logsTotal} 条</span>
        </Space>
        <Table
          rowKey="id"
          loading={logsLoading}
          dataSource={logs}
          size="small"
          pagination={{
            current: logsPage,
            pageSize: logsPageSize,
            total: logsTotal,
            showSizeChanger: true,
            onChange: (page, pageSize) => {
              if (!logRun) {
                return;
              }
              fetchLogs(logRun.id, page, pageSize);
            },
          }}
          columns={[
            {
              title: '序号',
              key: 'index',
              width: 70,
              render: (_: any, __: AiRunLog, index: number) =>
                (logsPage - 1) * logsPageSize + index + 1,
            },
            {
              title: '时间',
              dataIndex: 'timestamp',
              key: 'timestamp',
              width: 180,
              render: (v: string) =>
                v ? dayjs(v).format('YYYY-MM-DD HH:mm:ss') : '-',
            },
            {
              title: '级别',
              dataIndex: 'level',
              key: 'level',
              width: 80,
            },
            {
              title: '类别',
              dataIndex: 'category',
              key: 'category',
              width: 100,
            },
            {
              title: '消息',
              dataIndex: 'message',
              key: 'message',
              ellipsis: true,
            },
            {
              title: 'AI输入',
              key: 'ai_input',
              width: 160,
              render: (_: any, record: AiRunLog) =>
                record.ai_input ? JSON.stringify(record.ai_input) : '',
            },
            {
              title: 'AI输出',
              key: 'ai_output',
              width: 160,
              render: (_: any, record: AiRunLog) =>
                record.ai_output ? JSON.stringify(record.ai_output) : '',
            },
          ]}
          scroll={{ x: 900, y: 400 }}
        />
      </Modal>
    </div>
  );
};

export default AiInvestment;
