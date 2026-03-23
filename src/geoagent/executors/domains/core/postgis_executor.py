"""
PostGISExecutor - PostGIS 空间查询执行器
========================================
封装 PostGIS 空间数据库查询能力。

路由策略：
- GeoPandas + psycopg2（主力）
- SQLAlchemy + GeoAlchemy2（可选 ORM）

设计原则：全部 → 通过 Executor 调用，不让库互相调用
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from geoagent.executors.base import BaseExecutor, ExecutorResult


class PostGISExecutor(BaseExecutor):
    """
    PostGIS 空间查询执行器

    支持操作：
    - query：执行 SQL 查询
    - read：读取表/视图
    - write：写入表
    - spatial_query：空间查询（ST_Within, ST_Intersects, ST_DWithin）

    引擎：GeoPandas + psycopg2（主力）
    """

    task_type = "postgis"
    supported_engines = {"psycopg2", "sqlalchemy"}

    def run(self, task: Dict[str, Any]) -> ExecutorResult:
        operation = task.get("operation", "query")
        conn_str = task.get("connection_string", "")
        engine = task.get("engine", "psycopg2")

        if not conn_str:
            return ExecutorResult.err(
                self.task_type,
                "连接字符串不能为空（请提供 postgresql://user:pass@host:port/dbname）",
                engine="postgis"
            )

        if engine in ("psycopg2", "auto"):
            result = self._run_psycopg2(task)
            if result.success or engine == "psycopg2":
                return result
            result_sa = self._run_sqlalchemy(task)
            if result_sa.success:
                result_sa.warnings.append(f"psycopg2 失败，降级到 SQLAlchemy: {result.error}")
            return result_sa
        elif engine == "sqlalchemy":
            return self._run_sqlalchemy(task)
        else:
            return ExecutorResult.err(self.task_type, f"不支持的引擎: {engine}", engine=engine)

    def _run_psycopg2(self, task: Dict[str, Any]) -> ExecutorResult:
        """psycopg2 引擎（主力）"""
        try:
            import psycopg2
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "psycopg2 或 geopandas 不可用，请运行: pip install psycopg2-binary geopandas",
                engine="psycopg2"
            )

        try:
            import psycopg2
            import geopandas as gpd

            conn_str = task["connection_string"]
            operation = task.get("operation", "query")
            conn = psycopg2.connect(conn_str)

            if operation == "query":
                return self._psycopg2_query(conn, task)
            elif operation == "read":
                return self._psycopg2_read(conn, task)
            elif operation == "write":
                return self._psycopg2_write(conn, task)
            elif operation == "spatial_query":
                return self._psycopg2_spatial_query(conn, task)
            else:
                return ExecutorResult.err(
                    self.task_type,
                    f"不支持的操作: {operation}",
                    engine="psycopg2"
                )

        except psycopg2.OperationalError as e:
            return ExecutorResult.err(
                self.task_type,
                f"数据库连接失败: {str(e)}",
                engine="psycopg2"
            )
        except Exception as e:
            return ExecutorResult.err(
                self.task_type,
                f"PostGIS 操作失败: {str(e)}",
                engine="psycopg2"
            )

    def _psycopg2_query(self, conn, task: Dict[str, Any]) -> ExecutorResult:
        """执行 SQL 查询"""
        sql = task.get("sql", "")
        if not sql:
            return ExecutorResult.err(self.task_type, "SQL 查询语句不能为空", engine="psycopg2")

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                data = [dict(zip(columns, row)) for row in rows]

            return ExecutorResult.ok(
                self.task_type,
                "psycopg2",
                {
                    "operation": "query",
                    "row_count": len(data),
                    "columns": columns,
                    "rows": data[:1000],
                },
                meta={"sql": sql[:200]}
            )

        except Exception as e:
            return ExecutorResult.err(self.task_type, f"查询失败: {str(e)}", engine="psycopg2")

    def _psycopg2_read(self, conn, task: Dict[str, Any]) -> ExecutorResult:
        """读取表/视图"""
        import geopandas as gpd

        table = task.get("table", "")
        geom_col = task.get("geometry_column", "geom")
        output_file = task.get("output_file")
        where = task.get("where", "")

        if not table:
            return ExecutorResult.err(self.task_type, "表名不能为空", engine="psycopg2")
        if not output_file:
            return ExecutorResult.err(self.task_type, "输出文件路径不能为空", engine="psycopg2")

        try:
            sql = f"SELECT * FROM {table}"
            if where:
                sql += f" WHERE {where}"

            gdf = gpd.read_postgis(sql, conn, geom_col=geom_col)
            output_path = self._resolve_path(output_file)

            # 使用统一的保存方法，自动打包为ZIP
            actual_path, driver = self.save_geodataframe(gdf, output_path)

            return ExecutorResult.ok(
                self.task_type,
                "psycopg2",
                {
                    "operation": "read",
                    "table": table,
                    "feature_count": len(gdf),
                    "crs": str(gdf.crs) if gdf.crs else "unknown",
                    "columns": list(gdf.columns),
                    "output_file": actual_path,
                    "output_path": actual_path,
                },
                meta={"geometry_column": geom_col, "driver": driver}
            )

        except Exception as e:
            return ExecutorResult.err(self.task_type, f"读取表失败: {str(e)}", engine="psycopg2")

    def _psycopg2_write(self, conn, task: Dict[str, Any]) -> ExecutorResult:
        """写入表"""
        import geopandas as gpd

        input_file = task.get("input_file", "")
        table_name = task.get("table", "geoagent_output")
        if_exists = task.get("if_exists", "fail")

        if not input_file:
            return ExecutorResult.err(self.task_type, "输入文件不能为空", engine="psycopg2")

        try:
            gdf = gpd.read_file(self._resolve_path(input_file))
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)

            engine_url = task["connection_string"].replace("postgresql://", "postgresql+psycopg2://")
            try:
                from sqlalchemy import create_engine
                sa_engine = create_engine(engine_url)
                gdf.to_postgis(table_name, sa_engine, if_exists=if_exists, index=False)
                sa_engine.dispose()
            except ImportError:
                return ExecutorResult.err(
                    self.task_type,
                    "写入需要 SQLAlchemy，请运行: pip install sqlalchemy",
                    engine="psycopg2"
                )

            return ExecutorResult.ok(
                self.task_type,
                "psycopg2",
                {
                    "operation": "write",
                    "table": table_name,
                    "input_file": input_file,
                    "feature_count": len(gdf),
                    "if_exists": if_exists,
                },
                meta={"engine_used": "GeoPandas.to_postgis + SQLAlchemy"}
            )

        except Exception as e:
            return ExecutorResult.err(self.task_type, f"写入表失败: {str(e)}", engine="psycopg2")

    def _psycopg2_spatial_query(self, conn, task: Dict[str, Any]) -> ExecutorResult:
        """空间查询"""
        import geopandas as gpd

        source_table = task.get("source_table", "")
        filter_geom = task.get("filter_geometry", "")
        predicate = task.get("predicate", "ST_Intersects")
        output_file = task.get("output_file")
        buffer_dist = task.get("buffer_distance")

        if not source_table:
            return ExecutorResult.err(self.task_type, "源表名不能为空", engine="psycopg2")
        if not output_file:
            return ExecutorResult.err(self.task_type, "输出文件路径不能为空", engine="psycopg2")

        try:
            if filter_geom:
                if filter_geom.startswith("{"):
                    filter_geom = f"ST_GeomFromGeoJSON('{filter_geom}')"
                else:
                    filter_geom = f"ST_GeomFromText('{filter_geom}', 4326)"

                if predicate == "ST_DWithin" and buffer_dist:
                    sql = f"SELECT * FROM {source_table} WHERE ST_DWithin(geom, {filter_geom}, {buffer_dist})"
                else:
                    sql = f"SELECT * FROM {source_table} WHERE {predicate}(geom, {filter_geom})"
            else:
                sql = f"SELECT * FROM {source_table}"

            gdf = gpd.read_postgis(sql, conn, geom_col="geom")
            output_path = self._resolve_path(output_file)
            # 使用统一的保存方法，自动打包为ZIP
            actual_path, driver = self.save_geodataframe(gdf, output_path)

            return ExecutorResult.ok(
                self.task_type,
                "psycopg2",
                {
                    "operation": "spatial_query",
                    "source_table": source_table,
                    "predicate": predicate,
                    "feature_count": len(gdf),
                    "output_file": actual_path,
                    "output_path": actual_path,
                },
                meta={"engine_used": "PostGIS ST_* functions", "driver": driver}
            )

        except Exception as e:
            return ExecutorResult.err(self.task_type, f"空间查询失败: {str(e)}", engine="psycopg2")

    def _run_sqlalchemy(self, task: Dict[str, Any]) -> ExecutorResult:
        """SQLAlchemy 引擎（可选）"""
        try:
            from sqlalchemy import create_engine, text
            import geopandas as gpd
        except ImportError:
            return ExecutorResult.err(
                self.task_type,
                "SQLAlchemy 不可用，请运行: pip install sqlalchemy geoalchemy2",
                engine="sqlalchemy"
            )

        try:
            from sqlalchemy import create_engine, text
            import geopandas as gpd

            conn_str = task["connection_string"].replace("postgresql://", "postgresql+psycopg2://")
            engine = create_engine(conn_str)

            if task.get("operation") == "query":
                sql = task.get("sql", "SELECT 1")
                with engine.connect() as conn:
                    result = conn.execute(text(sql))
                    rows = [dict(row._mapping) for row in result]
                return ExecutorResult.ok(
                    self.task_type,
                    "sqlalchemy",
                    {"operation": "query", "rows": rows[:1000]},
                    meta={"sql": sql[:200]}
                )
            else:
                return self._run_psycopg2(task)

        except Exception as e:
            return ExecutorResult.err(self.task_type, f"SQLAlchemy 操作失败: {str(e)}", engine="sqlalchemy")
